import jax
import jax.numpy as jnp
import torch


def cv_embed(state, hx):
    w1 = state['embedding.weight']
    w2 = hx.param['tacotron2/~/embed']
    print("Embed", w2['embeddings'].shape)
    w2['embeddings'][:] = w1.numpy()


def cv_prenet(state, hx):
    cv_linear('decoder.prenet.layers.0.linear_layer.%s',
              'tacotron2/~/decoder/~/prenet/~/linear_norm/~/linear', state, hx,
              False)
    cv_linear('decoder.prenet.layers.1.linear_layer.%s',
              'tacotron2/~/decoder/~/prenet/~/linear_norm_1/~/linear', state,
              hx, False)


def cv_conv(name1, name2, state, hx, bias=True):
    w = state[name1 % "weight"].numpy().T
    print(name2, w.shape)
    hx.param[name2]['w'][:] = w
    if bias:
        b = state[name1 % "bias"].numpy()
        hx.param[name2]['b'][:] = jnp.expand_dims(b, -1)


def cv_lstm(name1, name2, state, hx):
    wih = state[name1 % "weight_ih"].numpy()
    whh = state[name1 % "weight_hh"].numpy()
    bih = state[name1 % "bias_ih"].numpy()
    bhh = state[name1 % "bias_hh"].numpy()

    wh = jnp.split(jnp.concatenate((wih, whh), -1).T, 4, axis=-1)
    bh = jnp.split(bih + bhh, 4, axis=-1)

    # pytorch matrix: [i f g o] is different from haiku matrix: [i g f+1 o]
    # so we need to reorder f and g, and decrease f's bias by 1.
    wh[1], wh[2] = wh[2], wh[1]
    bh[1], bh[2] = bh[2], bh[1]
    bh[2] -= 1
    wh = jnp.concatenate(wh, -1)
    bh = jnp.concatenate(bh, -1)

    print(name2, hx.param[name2]['b'].shape, hx.param[name2]['w'].shape)
    hx.param[name2]['b'][:] = bh
    hx.param[name2]['w'][:] = wh


def cv_batchnorm(name1, name2, state, hx):
    scale = state[name1 % "weight"].numpy().reshape((1, -1, 1))
    offset = state[name1 % "bias"].numpy().reshape((1, -1, 1))
    mean = state[name1 % "running_mean"].numpy().reshape((1, -1, 1))
    var = state[name1 % "running_var"].numpy().reshape((1, -1, 1))
    counter = state[name1 % "num_batches_tracked"].numpy().reshape(())

    hx.param[name2]['scale'][:] = scale
    hx.param[name2]['offset'][:] = offset
    hx.state[name2 + "/~/mean_ema"]['counter'].fill(counter)
    hx.state[name2 + "/~/var_ema"]['counter'].fill(counter)

    hx.state[name2 + "/~/mean_ema"]['average'][:] = mean
    hx.state[name2 + "/~/var_ema"]['average'][:] = var

    hx.state[name2 + "/~/mean_ema"]['hidden'][:] = mean
    hx.state[name2 + "/~/var_ema"]['hidden'][:] = var

    print(name2, scale.shape, offset.shape, mean.shape, var.shape)


def cv_linear(name1, name2, state, hx, bias=True):
    w = state[name1 % "weight"].numpy().T
    hx.param[name2]['w'][:] = w
    print(name2, w.shape)
    if bias:
        b = state[name1 % "bias"].numpy()
        hx.param[name2]['b'][:] = b


def to_haiku_model(torch_model, hparams):
    from hk_trainer import Trainer
    import numpy as onp

    trainer = Trainer(config=hparams)
    trainer.create_model()
    hx = jax.tree_map(lambda x: onp.copy(x[0]), trainer._hx)
    state = torch_model['state_dict']
    step = state["postnet.convolutions.0.1.num_batches_tracked"].item()

    cv_embed(state, hx)

    cv_prenet(state, hx)

    cv_lstm("encoder.lstm.%s_l0",
            "tacotron2/~/encoder/~/bi_lstm/~/lstm/linear", state, hx)
    cv_lstm("encoder.lstm.%s_l0_reverse",
            "tacotron2/~/encoder/~/bi_lstm/~/lstm_1/linear", state, hx)

    cv_lstm("decoder.attention_rnn.%s", "tacotron2/~/decoder/~/lstm/linear",
            state, hx)
    cv_lstm("decoder.decoder_rnn.%s", "tacotron2/~/decoder/~/lstm_1/linear",
            state, hx)

    cv_linear("decoder.linear_projection.linear_layer.%s",
              "tacotron2/~/decoder/~/linear_norm/~/linear", state, hx)
    cv_linear("decoder.gate_layer.linear_layer.%s",
              "tacotron2/~/decoder/~/linear_norm_1/~/linear", state, hx)

    cv_linear("decoder.attention_layer.query_layer.linear_layer.%s",
              "tacotron2/~/decoder/~/attention/~/linear_norm/~/linear", state,
              hx, False)
    cv_linear("decoder.attention_layer.memory_layer.linear_layer.%s",
              "tacotron2/~/decoder/~/attention/~/linear_norm_1/~/linear",
              state, hx, False)
    cv_linear("decoder.attention_layer.v.linear_layer.%s",
              "tacotron2/~/decoder/~/attention/~/linear_norm_2/~/linear",
              state, hx, False)
    cv_linear(
        "decoder.attention_layer.location_layer.location_dense.linear_layer.%s",
        "tacotron2/~/decoder/~/attention/~/location_layer/~/linear_norm/~/linear",
        state, hx, False)

    cv_conv("encoder.convolutions.0.0.conv.%s",
            "tacotron2/~/encoder/~/conv_norm/~/conv1_d", state, hx)
    cv_conv("encoder.convolutions.1.0.conv.%s",
            "tacotron2/~/encoder/~/conv_norm_1/~/conv1_d", state, hx)
    cv_conv("encoder.convolutions.2.0.conv.%s",
            "tacotron2/~/encoder/~/conv_norm_2/~/conv1_d", state, hx)

    cv_conv("postnet.convolutions.0.0.conv.%s",
            "tacotron2/~/postnet/~/conv_norm/~/conv1_d", state, hx)
    cv_conv("postnet.convolutions.1.0.conv.%s",
            "tacotron2/~/postnet/~/conv_norm_1/~/conv1_d", state, hx)
    cv_conv("postnet.convolutions.2.0.conv.%s",
            "tacotron2/~/postnet/~/conv_norm_2/~/conv1_d", state, hx)
    cv_conv("postnet.convolutions.3.0.conv.%s",
            "tacotron2/~/postnet/~/conv_norm_3/~/conv1_d", state, hx)
    cv_conv("postnet.convolutions.4.0.conv.%s",
            "tacotron2/~/postnet/~/conv_norm_4/~/conv1_d", state, hx)

    cv_conv(
        "decoder.attention_layer.location_layer.location_conv.conv.%s",
        "tacotron2/~/decoder/~/attention/~/location_layer/~/conv_norm/~/conv1_d",
        state, hx, False)

    cv_batchnorm("encoder.convolutions.0.1.%s",
                 "tacotron2/~/encoder/~/sequential/batch_norm", state, hx)
    cv_batchnorm("encoder.convolutions.1.1.%s",
                 "tacotron2/~/encoder/~/sequential_1/batch_norm", state, hx)
    cv_batchnorm("encoder.convolutions.2.1.%s",
                 "tacotron2/~/encoder/~/sequential_2/batch_norm", state, hx)

    cv_batchnorm("postnet.convolutions.0.1.%s",
                 "tacotron2/~/postnet/~/sequential/batch_norm", state, hx)
    cv_batchnorm("postnet.convolutions.1.1.%s",
                 "tacotron2/~/postnet/~/sequential_1/batch_norm", state, hx)
    cv_batchnorm("postnet.convolutions.2.1.%s",
                 "tacotron2/~/postnet/~/sequential_2/batch_norm", state, hx)
    cv_batchnorm("postnet.convolutions.3.1.%s",
                 "tacotron2/~/postnet/~/sequential_3/batch_norm", state, hx)
    cv_batchnorm("postnet.convolutions.4.1.%s",
                 "tacotron2/~/postnet/~/sequential_4/batch_norm", state, hx)

    return step, trainer._rng, hx


def main(pt, hk):
    from hparams import create_hparams
    hparams = create_hparams()
    print("Loading model from", pt)
    ck = torch.load(pt, map_location=torch.device("cpu"))
    step, rng, hx = to_haiku_model(ck, hparams)

    print("Saving model to", hk)
    torch.save((step, hparams.learning_rate, rng, hx), hk)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
