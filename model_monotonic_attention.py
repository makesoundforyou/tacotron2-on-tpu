from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from model import Prenet, Decoder, Tacotron2


def scale_gradient(x, s=1e-1):
    return x*s + x.data*(1.-s)


class MonoAttention(nn.Module):
    def __init__(self, num_mixtures, attention_rnn_dim, embedding_dim, attention_dim):
        super(MonoAttention, self).__init__()

        self._num_mixtures = num_mixtures
        self.F = nn.Sequential(
            nn.Linear(attention_rnn_dim, attention_dim, bias=True),
            nn.LayerNorm(attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 3*num_mixtures, bias=True),
            nn.LayerNorm(3*num_mixtures),
        )

        self.score_mask_value = 0
        self.register_buffer('pos', torch.arange(
            0, 10000, dtype=torch.float).view(1, -1, 1).data)

    def get_alignment_energies(self, attention_hidden_state, memory, previous_location):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        memory: encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        _t = self.F(attention_hidden_state.unsqueeze(1))
        w, delta, scale = _t.chunk(3, dim=-1)

        delta = torch.sigmoid(delta)
        loc = previous_location + delta
        std = torch.nn.functional.softplus(scale + 10)

        pos = self.pos[:, :memory.shape[1], :]
        z1 = torch.erf((loc-pos+0.5) / std)
        z2 = torch.erf((loc-pos-0.5) / std)
        z = (z1 - z2)*0.5
        w = torch.softmax(w, dim=-1)
        z = torch.bmm(z, w.squeeze(1).unsqueeze(2)).squeeze(-1)

        return z, loc

    def forward(self, attention_hidden_state, memory, previous_location, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment, loc = self.get_alignment_energies(
            attention_hidden_state, memory, previous_location)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = alignment  # F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, loc


class LSTMCellWithZoneout(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True,  zoneout_prob=0.1):
        super().__init__(input_size, hidden_size, bias)
        self._zoneout_prob = zoneout_prob

    def forward(self, input, hx):
        old_h, old_c = hx
        new_h, new_c = super(LSTMCellWithZoneout, self).forward(input, hx)
        if self.training:
            c_mask = torch.empty_like(new_c).bernoulli_(
                p=self._zoneout_prob).bool().data
            h_mask = torch.empty_like(new_h).bernoulli_(
                p=self._zoneout_prob).bool().data
            h = torch.where(h_mask, old_h, new_h)
            c = torch.where(c_mask, old_c, new_c)
            return h, c
        else:
            return new_h, new_c


class MonotonicDecoder(Decoder):
    def __init__(self, hparams):
        super(MonotonicDecoder, self).__init__(hparams)

        self.rnn1 = LSTMCellWithZoneout(
            hparams.prenet_dim + hparams.encoder_embedding_dim, hparams.attention_rnn_dim)

        self.rnn2 = LSTMCellWithZoneout(
            hparams.attention_rnn_dim, hparams.decoder_rnn_dim)

        self.attention_layer = MonoAttention(
            hparams.num_att_mixtures,
            hparams.decoder_rnn_dim, hparams.encoder_embedding_dim, hparams.attention_dim)

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.rnn1_hx = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.rnn1_cx = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.rnn2_hx = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.rnn2_cx = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.previous_location = Variable(memory.data.new(B, 1, 1).zero_())
        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(
            memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(
            memory.data.new(B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.mask = mask

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
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        gate_outputs = None
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
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

        self.attention_context, self.attention_weights, self.previous_location = self.attention_layer(
            self.rnn2_hx, self.memory, self.previous_location, self.mask)

        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.rnn1_hx, self.rnn1_cx = self.rnn1(
            cell_input, (self.rnn1_hx, self.rnn1_cx))
        self.rnn2_hx, self.rnn2_cx = self.rnn2(
            self.rnn1_hx, (self.rnn2_hx, self.rnn2_cx))
        decoder_hidden_attention_context = torch.cat(
            (self.rnn2_hx, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = None
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
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

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [attention_weights]

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

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class MonotonicTacotron2(Tacotron2):
    def __init__(self, hparams):
        super(MonotonicTacotron2, self).__init__(hparams)
        self.decoder = MonotonicDecoder(hparams)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)

        return outputs
