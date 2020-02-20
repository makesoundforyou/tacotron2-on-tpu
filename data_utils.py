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
        self._mean = torch.tensor([[-2.6924, -2.9219, -2.6817, -2.2255, -1.9451, -2.0005, -2.7405, -3.4584,
        -3.5648, -3.5789, -3.7116, -3.9104, -4.1431, -4.2432, -4.2828, -4.6060,
        -4.6829, -4.7948, -4.9358, -5.1079, -5.2964, -5.2176, -5.3302, -5.3744,
        -5.4201, -5.4619, -5.4922, -5.5530, -5.6502, -5.7646, -5.9569, -6.1800,
        -6.3441, -6.4399, -6.4922, -6.2843, -6.2767, -6.2586, -6.2419, -6.5388,
        -6.2956, -6.4274, -6.3946, -6.4998, -6.5961, -6.5975, -6.6392, -6.7964,
        -6.9023, -7.0195, -7.1859, -7.2549, -7.2613, -7.1978, -7.3244, -7.4186,
        -7.5808, -7.5587, -7.5464, -7.6480, -7.8024, -7.9106, -7.9199, -8.0032,
        -8.0470, -8.0612, -8.3861, -8.7617, -8.6546, -8.3202, -8.0038, -7.9196,
        -7.9125, -7.8028, -7.6964, -7.5929, -7.4815, -7.4634, -7.5408, -7.7315]]).float().t()
        self._std = torch.tensor([[1.2258, 1.3116, 1.8001, 2.3640, 2.7114, 2.8374, 2.6605, 2.4154, 2.4972,
        2.5725, 2.6020, 2.5795, 2.5657, 2.5478, 2.4319, 2.3029, 2.2750, 2.2774,
        2.2849, 2.2990, 2.3300, 2.3444, 2.3227, 2.2612, 2.2097, 2.1690, 2.1477,
        2.1438, 2.1394, 2.1112, 2.0694, 2.0333, 1.9991, 1.9708, 1.9616, 1.9120,
        1.7027, 1.6323, 1.8462, 1.9795, 2.0024, 2.0136, 2.0167, 2.0032, 1.9770,
        1.9665, 1.9534, 1.9111, 1.8302, 1.8358, 1.8805, 1.9084, 1.9131, 1.9199,
        1.7860, 1.6244, 1.8035, 1.7886, 1.8283, 1.8300, 1.7908, 1.7415, 1.7295,
        1.7082, 1.6775, 1.6285, 1.7002, 1.7588, 1.7188, 1.7752, 1.8050, 1.8515,
        1.8588, 1.8044, 1.8547, 1.8919, 1.9182, 1.9493, 1.9738, 1.9258]]).float().t()

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
        if self.len >= 900: self.len = 900
        print("data len ", self.len)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        
        if audiopath in self.mm:
            mel = self.mm[audiopath]
        else:
            mel = self.get_mel(audiopath)
            self.mm[audiopath] = mel
        
        return (text, mel[:, :self.len].sub(self._mean).div(self._std))

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
