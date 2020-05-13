from collections import namedtuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import torch
from jax.experimental import optix
from jax.interpreters.pxla import replicate as rep

from hk_model import Tacotron2

NetState = namedtuple("NetState", "param state")
TrainerState = namedtuple("TrainerState", "param state opt_state")


def replicate(x):
    n = jax.device_count()
    return jax.tree_map(lambda a: rep(a, n, n), x)


def treemap_first_elem(x):
    """The inverse of `replicate` function
    """
    return jax.tree_map(lambda a: a[0], x)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, text_len=20, mel_len=100):
        # we only support one frame per step
        assert (n_frames_per_step == 1)
        self.n_frames_per_step = n_frames_per_step
        self.text_len = text_len
        self.mel_len = mel_len

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        text_len = max([len(x[0]) for x in batch]) // 5 * 5 + 5
        self.text_len = max(self.text_len, text_len)

        text_padded = torch.LongTensor(len(batch), self.text_len)
        text_padded.zero_()
        input_lengths = torch.LongTensor([len(x[0]) for x in batch])
        for i in range(len(batch)):
            text = batch[i][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        mel_len = max([x[1].size(1) for x in batch]) // 5 * 5 + 5
        self.mel_len = max(self.mel_len, mel_len)

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, self.mel_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), self.mel_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(batch)):
            mel = batch[i][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths


def to_mask(ls, maxlen):
    mask = np.zeros((len(ls), maxlen))
    for i, le in enumerate(ls):
        mask[i, 0:le] = 1
    return mask


def loss_fn(output, target):
    def bce(x, z):
        """Binary cross entropy loss
        note that sigmoid(-x) = 1-sigmoid(x)
        return -log(p)   if z=1,
           and -log(1-p) if z=0
        """
        loss = -jax.nn.log_sigmoid(x) * z - jax.nn.log_sigmoid(-x) * (1 - z)
        return jnp.mean(loss)

    def mse(x, y):
        return jnp.mean(jnp.square(x - y))

    mel_target, gate_target = target
    mel_out, mel_out_postnet, gate_out, _ = output
    l1 = mse(mel_out, mel_target)
    l2 = mse(mel_out_postnet, mel_target)
    l3 = bce(gate_out, gate_target)
    return l1 + l2 + l3


def forward(x, config):
    net = Tacotron2(config)
    out = net(x)
    return out


# we use currying to remove `config` from function calls
@jax.curry
def loss_forward(config, x, y):
    out = forward(x, config)
    return loss_fn(out, y)


class ClipByGlobalNormState(optix.OptState):
    """The `clip_by_global_norm` transformation is stateless."""


def clip_by_global_norm(max_norm) -> optix.InitUpdate:
    def init_fn(_):
        return ClipByGlobalNormState()

    def update_fn(updates, state):
        g_norm = optix.global_norm(updates) + 1e-6
        scale = jnp.clip(max_norm / g_norm, a_max=1.)
        updates = jax.tree_map(lambda g: g * scale, updates)
        return updates, state

    return optix.InitUpdate(init_fn, update_fn)


class Trainer:
    def __init__(self, config):
        self.config = config

        self.optimizer = self.create_optimizer()
        self.compile_updater()

        self._step = 0
        self._hx = None
        self._rng = random.PRNGKey(config.seed)
        self.val_fn = None

    def validate(self, x, y):
        def net_validate(x, y):
            out = forward(x, self.config)
            loss = loss_fn(out, y)
            return loss, out

        # only transform once
        if self.val_fn is None:
            f = hk.transform_with_state(net_validate).apply
            self.val_fn = jax.pmap(f, axis_name='i')

        rngs = random.split(self.next_rng(), jax.device_count())
        loss, out = self.val_fn(*self._hx[:2], rngs, x, y)[0]
        return float(jnp.mean(loss)), out

    def inference(self, text):
        f = lambda txt: Tacotron2(self.config).inference(txt)
        f = hk.transform_with_state(f).apply
        hx = treemap_first_elem(self._hx[:2])
        return f(*hx, self.next_rng(), text)[0]

    def create_optimizer(self):
        def warmup_scheduler(init_step: int):
            """
            warm-up the first `init_step`
            """
            def fn(step):
                # reduce r1 by half for every 50_000 steps
                p = jnp.floor(step * 1.0 / 50_000.)
                p = jnp.clip(p, a_max=5.)
                r1 = jnp.power(2., -p)

                # increase r2 from 0. to 1. in `init_step`
                r2 = jnp.clip(step * 1.0 / init_step, a_max=1.0)
                return r1 * r2

            return fn

        return optix.chain(clip_by_global_norm(self.config.grad_clip_thresh),
                           optix.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                           optix.scale(-self.config.learning_rate),
                           optix.scale_by_schedule(warmup_scheduler(100)))

    def compile_updater(self):
        f = hk.transform_with_state(loss_forward(self.config)).apply
        vag = jax.value_and_grad(f, has_aux=True)

        def updater(hx: TrainerState, rng: jnp.ndarray, x, y):
            (v, state), grads = vag(*hx[:2], rng, x, y)
            grads = jax.lax.pmean(grads, axis_name='i')

            # the regularized loss function:
            # L(w) = 1/n x SUM_i l(xi, w) + 1/2 x weight_decay x ||w||^2
            grads = jax.tree_multimap(
                lambda g, p: g + p * self.config.weight_decay, grads, hx.param)

            gn = optix.global_norm(grads)
            grads, opt_state = self.optimizer.update(grads, hx.opt_state)
            param = optix.apply_updates(hx.param, grads)
            return (v, gn), TrainerState(param, state, opt_state)

        self.updater = jax.pmap(updater, axis_name='i')

    def step(self, x, y):
        rngs = random.split(self.next_rng(), jax.device_count())
        (f, gn), self._hx = self.updater(self._hx, rngs, x, y)
        self._step += 1
        return float(jnp.mean(f)), float(gn[0])

    def load_checkpoint(self, path):
        step, lr, rng, hx = torch.load(path)
        self._hx = replicate(hx)
        self._step = step
        self._rng = rng
        self.config.learning_rate = lr
        self.optimizer = self.create_optimizer()
        self.compile_updater()
        return step

    def to_device(self):
        self._hx = jax.device_put(self._hx)

    def save_checkpoint(self, path):
        hx = treemap_first_elem(self._hx)
        torch.save((self._step, self.config.learning_rate, self._rng, hx),
                   path)

    def next_rng(self):
        self._rng, rng = random.split(self._rng)
        return rng

    def warm_start_model(self, path, skip_layers):
        _, _, _, hx = torch.load(path)
        self._hx = treemap_first_elem(self._hx)  # collapse to a single device
        param = hk.data_structures.to_mutable_dict(self._hx.param)
        for k in param.keys():
            print(f"Layer:{k:>80} ", end="\t")
            if k not in skip_layers:
                # Copy weigths from the pretrained model
                param[k] = hx.param[k]
                print("  [ √ ]")
            else:
                print("  [ x ]")

        param = hk.data_structures.to_immutable_dict(param)
        # Copy network state and optimizer state from the pretrained model
        self._hx = TrainerState(param, hx.state, self._hx.opt_state)
        self._hx = replicate(self._hx)  # to multiple devices
        self.compile_updater()

    def create_model(self):
        """Create a new random model
        Weights are randomly initialized
        """
        init_fn = hk.transform_with_state(loss_forward(self.config)).init
        init_fn = jax.pmap(init_fn, axis_name='i')

        # dummy inputs
        mel = jnp.zeros((1, 80, 25))
        text = jnp.zeros((1, 20), dtype='int32')
        text_mask = to_mask([5], maxlen=20)
        mel_mask = to_mask([20], maxlen=25)
        x = (text, text_mask, mel, mel_mask)
        y = (mel, mel_mask)
        rxy = replicate((self.next_rng(), x, y))
        netstate = NetState(*init_fn(*rxy))
        opt_state = self.optimizer.init(treemap_first_elem(netstate.param))
        opt_state = replicate(opt_state)
        self._hx = TrainerState(netstate.param, netstate.state, opt_state)

    def parse_batch(self, batch):
        text_padded, input_lengths, \
                mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.int().numpy()
        input_lengths = input_lengths.int()
        max_len = text_padded.shape[1]
        input_lengths = input_lengths.numpy()
        text_mask = to_mask(input_lengths, max_len)
        mel_padded = mel_padded.float().numpy()
        gate_padded = gate_padded.float().numpy()
        output_lengths = output_lengths.int().numpy()
        ol = mel_padded.shape[2]
        mel_mask = to_mask(output_lengths, ol)

        x = text_padded, text_mask, mel_padded, mel_mask
        y = mel_padded, gate_padded
        n = jax.device_count()
        b = text_padded.shape[0]

        if b % n != 0:
            raise ValueError(
                f"batch size ({b}) is not divisible by number of devices ({n})"
            )

        def reshape(x):
            # [ [mini-batch1 for device 1],
            #   [mini-batch2 for device 2], ...]
            shape = (n, -1, *x.shape[1:])
            return x.reshape(shape)

        x, y = jax.tree_map(reshape, (x, y))
        return x, y
