import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, start_len=50, step=50):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._mean = torch.tensor([[0.8325, 0.8118, 0.8576, 0.8729, 0.8355, 0.7786, 0.7631, 0.7646, 0.7566,
        0.7343, 0.7257, 0.7283, 0.7103, 0.6922, 0.6823, 0.6733, 0.6599, 0.6458,
        0.6416, 0.6427, 0.6382, 0.6270, 0.6242, 0.6232, 0.6154, 0.6094, 0.5958,
        0.5720, 0.5566, 0.5468, 0.5451, 0.5472, 0.5631, 0.5644, 0.5507, 0.5506,
        0.5547, 0.5549, 0.5453, 0.5372, 0.5348, 0.5323, 0.5171, 0.5069, 0.4972,
        0.4828, 0.4758, 0.4823, 0.4757, 0.4654, 0.4480, 0.4485, 0.4508, 0.4435,
        0.4299, 0.4205, 0.4173, 0.4100, 0.4070, 0.3977, 0.3603, 0.3511, 0.3786,
        0.4093, 0.4202, 0.4212, 0.4294, 0.4397, 0.4490, 0.4587, 0.4557, 0.4455,
        0.4227, 0.4136, 0.4021, 0.3781, 0.3510, 0.3294, 0.3204, 0.2951]]).float().t()
        self._std = torch.tensor([[0.1166, 0.1366, 0.1793, 0.1948, 0.1920, 0.1823, 0.1881, 0.2014, 0.2039,
        0.2029, 0.2052, 0.2017, 0.1916, 0.1842, 0.1868, 0.1878, 0.1891, 0.1901,
        0.1915, 0.1902, 0.1875, 0.1833, 0.1799, 0.1778, 0.1783, 0.1774, 0.1739,
        0.1703, 0.1681, 0.1652, 0.1640, 0.1635, 0.1449, 0.1392, 0.1628, 0.1661,
        0.1686, 0.1691, 0.1686, 0.1671, 0.1667, 0.1658, 0.1628, 0.1552, 0.1560,
        0.1596, 0.1610, 0.1620, 0.1609, 0.1402, 0.1478, 0.1518, 0.1542, 0.1546,
        0.1511, 0.1471, 0.1466, 0.1435, 0.1418, 0.1400, 0.1495, 0.1491, 0.1504,
        0.1537, 0.1573, 0.1571, 0.1526, 0.1586, 0.1618, 0.1643, 0.1675, 0.1645,
        0.1590, 0.1514, 0.1443, 0.1319, 0.1292, 0.1299, 0.1264, 0.1240]]).float().t()

        self._start_len = start_len
        self.len = start_len-step
        self._step = step
        self.mm = {}

        # self.stats()

    def stats(self):
        m = 0.
        mm = 0.
        l = 0
        for a in self.audiopaths_and_text:
            # print(a)
            p, t = a
            # print(p, t)
            mel = self.get_mel(p)
            l = l + mel.shape[1]
            m = m + mel.sum(axis=-1)
            mm = mm + mel.pow(2).sum(axis=-1)
        m = m / l 
        mm = mm / l
        std = torch.sqrt(mm - m.pow(2))
        print(m)
        print(std)

    def step(self):
        self.len += self._step
        if self.len >= 1000: self.len = 1000

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        
        if audiopath in self.mm:
            mel = self.mm[audiopath]
        else:
            mel = self.get_mel(audiopath)
            self.mm[audiopath] = mel
        
        if self._start_len == -1:
            return (text, mel[:, 12:].sub(self._mean).div(self._std))
        else:
            return (text, mel[:, 12:self.len].sub(self._mean).div(self._std))

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
