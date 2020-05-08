from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from utils import to_gpu, get_mask_from_lengths
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import math
from haiku.initializers import Constant, TruncatedNormal, RandomUniform


def calculate_gain(fn: str) -> float:
    fn = fn.lower()
    dic = {
        'linear': 1.,
        'identity': 1.,
        'sigmoid': 1.,
        'tanh': 5. / 3.,
        'relu': math.sqrt(2)
    }
    if fn in dic:
        return dic[fn]
    else:
        raise NotImplementedError("Not supported function")


class LinearNorm(hk.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        val = calculate_gain(w_init_gain) * math.sqrt(6. / (in_dim + out_dim))
        self.linear_layer = hk.Linear(output_size=out_dim,
                                      with_bias=bias,
                                      w_init=RandomUniform(-val, val))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear_layer(x)


class ConvNorm(hk.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True,
                 w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding_ = int(dilation * (kernel_size - 1) / 2)
        else:
            padding_ = int(dilation * (kernel_size - 1) / 2)
            assert (padding == padding_)

        gain = calculate_gain(w_init_gain)
        assert (kernel_size == 5 or kernel_size == 31)
        fanin = in_channels * kernel_size
        fanout = out_channels * kernel_size
        val = gain * math.sqrt(6. / (fanin + fanout))
        self.conv = hk.Conv1D(output_channels=out_channels,
                              kernel_shape=[kernel_size],
                              stride=stride,
                              rate=dilation,
                              with_bias=bias,
                              data_format="NCW",
                              padding='SAME',
                              w_init=RandomUniform(-val, val))

        # self.conv = torch.nn.Conv1d(in_channels, out_channels,
        #                             kernel_size=kernel_size, stride=stride,
        #                             padding=padding, dilation=dilation,
        #                             bias=bias)

        # torch.nn.init.xavier_uniform_(
        #     self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def __call__(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class BiLSTM(hk.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.lstm1 = hk.LSTM(hidden_size=hidden_size)
        self.lstm2 = hk.LSTM(hidden_size=hidden_size)

    def __call__(self, x):
        bs = x.shape[0]
        x = jnp.swapaxes(x, 0, 1)

        hx1 = self.lstm1.initial_state(bs)
        hx2 = self.lstm2.initial_state(bs)

        x1, _ = hk.dynamic_unroll(self.lstm1, x, hx1)
        x_rev = jnp.flip(x, axis=0)
        x2_rev, _ = hk.dynamic_unroll(self.lstm2, x_rev, hx2)
        x2 = jnp.flip(x2_rev, axis=0)
        x = jnp.concatenate((x1, x2), axis=-1)
        return jnp.swapaxes(x, 0, 1)


def dropout(x, p, training, rng=None):
    rng = hk.next_rng_key() if rng is None else rng
    x = hk.dropout(rng, p, x) if training else x
    return x


###### START HERE ####


class LocationLayer(hk.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2,
                                      attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding,
                                      bias=False,
                                      stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters,
                                         attention_dim,
                                         bias=False,
                                         w_init_gain='tanh')

    def __call__(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = jnp.swapaxes(processed_attention, 1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(hk.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim,
                                      attention_dim,
                                      bias=False,
                                      w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim,
                                       attention_dim,
                                       bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(jnp.expand_dims(query, 1))
        processed_attention_weights = self.location_layer(
            attention_weights_cat)
        energies = self.v(
            jnp.tanh(processed_query + processed_attention_weights +
                     processed_memory))

        energies = jnp.squeeze(energies, -1)
        return energies

    def __call__(self, attention_hidden_state, memory, processed_memory,
                 attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(attention_hidden_state,
                                                processed_memory,
                                                attention_weights_cat)

        if mask is not None:
            alignment = jnp.where(mask, alignment,
                                  self.score_mask_value)  ### - ( - mask)
            """
            alignment.data.masked_fill_(mask, self.score_mask_value)
            """

        attention_weights = jax.nn.softmax(alignment, axis=1)
        attention_context = jax.lax.batch_matmul(
            jnp.expand_dims(attention_weights, 1), memory)
        attention_context = jnp.squeeze(attention_context, 1)

        return attention_context, attention_weights


class Prenet(hk.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = [
            LinearNorm(in_size, out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ]

    def __call__(self, x):
        for linear in self.layers:
            x = dropout(jax.nn.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(hk.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = []
        self.training = hparams.is_training

        def batchnorm(x):
            bn = hk.BatchNorm(True, True, 0.9, data_format="NCW")
            return bn(x, self.training)

        self.convolutions.append(
            hk.Sequential([
                ConvNorm(hparams.n_mel_channels,
                         hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size,
                         stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='tanh'), batchnorm
            ]))

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                hk.Sequential([
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size,
                             stride=1,
                             padding=int(
                                 (hparams.postnet_kernel_size - 1) / 2),
                             dilation=1,
                             w_init_gain='tanh'), batchnorm
                ]))

        self.convolutions.append(
            hk.Sequential([
                ConvNorm(hparams.postnet_embedding_dim,
                         hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size,
                         stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='linear'), batchnorm
            ]))

    def __call__(self, x):
        for i in range(len(self.convolutions) - 1):
            x = dropout(jnp.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = dropout(self.convolutions[-1](x), 0.5, self.training)

        return x

    def inference(self, x):
        for i in range(len(self.convolutions) - 1):
            x = dropout(jnp.tanh(self.convolutions[i](x)), 0.5, False)
        x = dropout(self.convolutions[-1](x), 0.5, False)
        return x


class Encoder(hk.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.training = hparams.is_training

        def batchnorm(x):
            bn = hk.BatchNorm(True, True, 0.9, data_format="NCW")
            return bn(x, self.training)

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = hk.Sequential([
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size,
                         stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='relu'), batchnorm
            ])
            convolutions.append(conv_layer)
        self.convolutions = convolutions

        self.lstm = BiLSTM(hparams.encoder_embedding_dim // 2)
        """
        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                    int(hparams.encoder_embedding_dim / 2),
                    1,
                    batch_first=True,
                    bidirectional=True)
        """

    def __call__(self, x, mask):
        for conv in self.convolutions:
            x = dropout(jax.nn.relu(conv(x)), 0.5, self.training)

        x = jnp.swapaxes(x, 1, 2)
        """
        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              input_lengths,
                                              batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)

        return outputs
        """
        mask = jnp.expand_dims(mask, -1)
        output = jnp.where(mask, self.lstm(x), 0.0)
        return output

    def inference(self, x):
        for conv in self.convolutions:
            x = dropout(jax.nn.relu(conv(x)), 0.5, False)

        x = jnp.swapaxes(x, 1, 2)

        outputs = self.lstm(x)

        return outputs


class Decoder(hk.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.training = hparams.is_training

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = hk.LSTM(hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = hk.LSTM(hparams.decoder_rnn_dim)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(hparams.decoder_rnn_dim +
                                     hparams.encoder_embedding_dim,
                                     1,
                                     bias=True,
                                     w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.shape[0]
        decoder_input = jnp.zeros(
            (B, self.n_mel_channels * self.n_frames_per_step),
            dtype=memory.dtype)
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.shape[0]
        MAX_TIME = memory.shape[1]

        t = memory.dtype

        attention_hidden = jnp.zeros((B, self.attention_rnn_dim), dtype=t)
        attention_cell = jnp.zeros((B, self.attention_rnn_dim), dtype=t)

        decoder_hidden = jnp.zeros((B, self.decoder_rnn_dim), dtype=t)
        decoder_cell = jnp.zeros((B, self.decoder_rnn_dim), dtype=t)

        attention_weights = jnp.zeros((B, MAX_TIME), dtype=t)
        attention_weights_cum = jnp.zeros((B, MAX_TIME), dtype=t)
        attention_context = jnp.zeros((B, self.encoder_embedding_dim), dtype=t)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        return (attention_hidden,
                attention_cell), (attention_context, attention_weights,
                                  attention_weights_cum), (decoder_hidden,
                                                           decoder_cell)

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = jnp.swapaxes(decoder_inputs, 1, 2)
        decoder_inputs = decoder_inputs.reshape(
            (decoder_inputs.shape[0],
             int(decoder_inputs.shape[1] / self.n_frames_per_step), -1))
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = jnp.swapaxes(decoder_inputs, 0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out) where B = 1
        # alignments = jnp.swapaxes(alignments, 0, 1)
        alignments = jnp.expand_dims(alignments, 0)
        # (T_out, B) -> (B, T_out)
        gate_outputs = jnp.swapaxes(gate_outputs, 0, 1)
        gate_outputs = jnp.squeeze(gate_outputs, -1)
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = jnp.swapaxes(mel_outputs, 0, 1)
        # .transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.reshape(
            (mel_outputs.shape[0], -1, self.n_mel_channels))
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = jnp.swapaxes(mel_outputs, 1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, input, state):
        (self_attention_hidden, self_attention_cell), (
            self_attention_context, self_attention_weights,
            self_attention_weights_cum), (self_decoder_hidden,
                                          self_decoder_cell) = state
        rng1, rng2, decoder_input = input
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = jnp.concatenate((decoder_input, self_attention_context),
                                     -1)
        self_attention_hidden, self_attention_cell = self.attention_rnn(
            cell_input, (self_attention_hidden, self_attention_cell))[1]
        self_attention_hidden = dropout(self_attention_hidden,
                                        self.p_attention_dropout,
                                        self.training, rng1)

        attention_weights_cat = jnp.concatenate(
            (jnp.expand_dims(self_attention_weights, 1),
             jnp.expand_dims(self_attention_weights_cum, 1)),
            axis=1)
        self_attention_context, self_attention_weights = self.attention_layer(
            self_attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self_attention_weights_cum = self_attention_weights_cum + self_attention_weights
        decoder_input = jnp.concatenate(
            (self_attention_hidden, self_attention_context), axis=-1)
        self_decoder_hidden, self_decoder_cell = self.decoder_rnn(
            decoder_input, (self_decoder_hidden, self_decoder_cell))[1]
        self_decoder_hidden = dropout(self_decoder_hidden,
                                      self.p_decoder_dropout, self.training,
                                      rng2)

        decoder_hidden_attention_context = jnp.concatenate(
            (self_decoder_hidden, self_attention_context), axis=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        state = (self_attention_hidden, self_attention_cell), (
            self_attention_context, self_attention_weights,
            self_attention_weights_cum), (self_decoder_hidden,
                                          self_decoder_cell)
        return state, (decoder_output, gate_prediction,
                       self_attention_weights[0])

    def __call__(self, memory, decoder_inputs, text_mask):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = jnp.expand_dims(self.get_go_frame(memory), 0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = jnp.concatenate((decoder_input, decoder_inputs),
                                         axis=0)
        decoder_inputs = self.prenet(decoder_inputs)

        state = self.initialize_decoder_states(memory, mask=text_mask)

        mel_outputs, gate_outputs, alignments = [], [], []

        def scan_body(prev_state, next_input):
            return self.decode(next_input, prev_state)

        rng1s = random.split(hk.next_rng_key(), decoder_inputs.shape[0] - 1)
        rng2s = random.split(hk.next_rng_key(), decoder_inputs.shape[0] - 1)

        _, (mel_outputs, gate_outputs,
            alignments) = jax.lax.scan(scan_body, state,
                                       (rng1s, rng2s, decoder_inputs[:-1]))
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        state = self.initialize_decoder_states(memory, mask=None)
        rng1 = hk.next_rng_key()
        rng2 = hk.next_rng_key()

        mel_outputs, gate_outputs, alignments = [], [], []
        prenet_jit = hk.jit(lambda x: self.prenet(x))
        decode_jit = hk.jit(lambda x, y: self.decode(x, y))
        while True:
            decoder_input = prenet_jit(decoder_input)
            state, (mel_output, gate_output, alignment) = decode_jit(
                (rng1, rng2, decoder_input), state)

            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(alignment)

            if jnp.all(jax.nn.sigmoid(gate_output) > self.gate_threshold):
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs = jnp.stack(mel_outputs)
        gate_outputs = jnp.stack(gate_outputs)
        alignments = jnp.stack(alignments)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(hk.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding = hk.Embed(vocab_size=hparams.n_symbols,
                                  embed_dim=hparams.symbols_embedding_dim,
                                  w_init=RandomUniform(-val, val))
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return ((text_padded, input_lengths, mel_padded, max_len,
                 output_lengths), (mel_padded, gate_padded))

    def parse_output(self, outputs, mel_mask=None):
        if self.mask_padding and mel_mask is not None:
            """
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
            """
            mel_mask = jnp.expand_dims(mel_mask, 1)
            o0 = jnp.where(mel_mask, outputs[0], 0.0)
            o1 = jnp.where(mel_mask, outputs[1], 0.0)
            o2 = jnp.where(mel_mask[:, 0, :], outputs[2], 1e3)  # gate energies
            outputs = (o0, o1, o2, *outputs[3:])

        return outputs

    def __call__(self, inputs):
        text_inputs, text_mask, mels, mel_mask = inputs
        # text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = jnp.swapaxes(self.embedding(text_inputs), 1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_mask)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, text_mask=text_mask)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            mel_mask)

    def inference(self, text):
        embedded_inputs = jnp.swapaxes(self.embedding(text), 1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet.inference(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
