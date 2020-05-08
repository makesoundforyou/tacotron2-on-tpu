import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from hk_model import Tacotron2
import torch

from hparams import create_hparams
from collections import namedtuple
from jax.experimental import optix

import numpy as np
NetState = namedtuple("NetState", "param state")
TrainerState = namedtuple("TrainerState", "param state opt_state")


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, text_len=200, mel_len=870):
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
        # input_lengths, ids_sorted_decreasing = torch.sort(
        #     torch.LongTensor([len(x[0]) for x in batch]),
        #     dim=0, descending=True)
        # max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), self.text_len)
        text_padded.zero_()
        input_lengths = torch.LongTensor([len(x[0]) for x in batch])
        for i in range(len(batch)):
            text = batch[i][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = self.mel_len
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
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


def bce(x, z):
    l = -jax.nn.log_sigmoid(x) * z - jax.nn.log_sigmoid(-x) * (1 - z)
    return jnp.mean(l)


def mse(x, y):
    return jnp.mean(jnp.square(x - y))


def loss_fn(output, target):
    mel_target, gate_target = target
    mel_out, mel_out_postnet, gate_out, _ = output
    l1 = mse(mel_out, mel_target)
    l2 = mse(mel_out_postnet, mel_target)
    l3 = bce(gate_out, gate_target)
    return l1 + l2 + l3


def forward(x, config):
    text, text_mask, mel, mel_mask = x
    net = Tacotron2(config)
    inp = (text, text_mask, mel, mel_mask)
    out = net(inp)
    return out


def loss_forward(x, y, config):
    out = forward(x, config)
    return loss_fn(out, y)


def net_validate(x, y, config):
    out = forward(x, config)
    loss = loss_fn(out, y)
    return loss, out


from typing import Dict, Tuple, Sequence


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
        self.lr = config.learning_rate

        self.optimizer = optix.chain(
            clip_by_global_norm(config.grad_clip_thresh),
            optix.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
            optix.scale(-(self.lr)))

        init_fn, f = hk.transform_with_state(loss_forward)
        vag = jax.value_and_grad(f, has_aux=True)

        _, dm_val_fn = hk.transform_with_state(net_validate)

        def val_fn(hx: TrainerState, rng, x, y):
            return dm_val_fn(hx.param, hx.state, rng, x, y, config=self.config)

        self.val_fn = jax.jit(val_fn) if config.enable_jit else val_fn

        def updater(hx: TrainerState, rng: jnp.ndarray, x, y):
            (v, state), grads = vag(hx.param,
                                    hx.state,
                                    rng,
                                    x,
                                    y,
                                    config=config)
            grads = jax.tree_multimap(lambda g, p: g + p * config.weight_decay,
                                      grads, hx.param)

            gn = optix.global_norm(grads)
            grads, opt_state = self.optimizer.update(grads, hx.opt_state)
            param = optix.apply_updates(hx.param, grads)
            return (v, gn), TrainerState(param, state, opt_state)

        self.updater = jax.jit(updater) if config.enable_jit else updater
        self._step = 0
        self._hx = None
        self._rng = random.PRNGKey(config.seed)

    def validate(self, x, y):
        (loss, y_predicted), _ = self.val_fn(self._hx, self.next_rng(), x, y)
        return loss, y_predicted

    def step(self, x, y):
        (f, gn), hx = self.updater(self._hx, self.next_rng(), x, y)
        self._hx = hx
        self._step += 1
        return float(f), float(gn)

    def load_checkpoint(self, path):
        step, rng, hx = torch.load(path)
        self._hx = hx
        self._step = step
        self._rng = rng
        return step

    def save_checkpoint(self, path):
        torch.save((self._step, self._rng, self._hx), path)

    def next_rng(self):
        self._rng, rng = random.split(self._rng)
        return rng

    def create_model(self):
        init_fn, f = hk.transform_with_state(loss_forward)
        vag = jax.value_and_grad(f, has_aux=True)

        mel = jnp.zeros((1, 80, 25))
        text = jnp.zeros((1, 20), dtype='int32')
        text_mask = to_mask([5], maxlen=20)
        mel_mask = to_mask([20], maxlen=25)
        x = (text, text_mask, mel, mel_mask)
        y = (mel, mel_mask)

        netstate = NetState(
            *init_fn(self.next_rng(), x, y, config=self.config))
        opt_state = self.optimizer.init(netstate.param)
        self._hx = TrainerState(netstate.param, netstate.state, opt_state)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = text_padded.long().numpy()
        input_lengths = input_lengths.long()
        max_len = text_padded.shape[1]  # input_lengths.max().item()
        input_lengths = input_lengths.numpy()
        text_mask = to_mask(input_lengths, max_len)
        # max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.float().numpy()
        gate_padded = gate_padded.float().numpy()
        output_lengths = output_lengths.long().numpy()
        ol = mel_padded.shape[2]  # output_lengths.max().item()
        mel_mask = to_mask(output_lengths, ol)
        # print(max_len, ol)

        return ((text_padded, text_mask, mel_padded, mel_mask), (mel_padded,
                                                                 gate_padded))
