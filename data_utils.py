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
        self._mean = torch.tensor([[-7.0222], [-6.1906], [-5.1736], [-4.2412], [-3.7652], [-3.6533], [-3.6642], [-3.7249], [-3.7714], [-3.7709], [-3.6496], [-3.5707],
                [-3.5742], [-3.6369], [-3.7370], [-3.9888], [-4.1180], [-4.1938], [-4.3030], [-4.4620], [-4.6258], [-4.7973], [-5.0267], [-5.0906],
                [-5.1643], [-5.1518], [-5.2571], [-5.2868], [-5.3991], [-5.4988], [-5.5740], [-5.7033], [-5.7849], [-5.8197], [-5.9224], [-5.8171],
                [-5.7680], [-5.6486], [-5.5940], [-5.5730], [-5.5224], [-5.4793], [-5.5243], [-5.6329], [-5.7697], [-5.8886], [-5.9992], [-6.0405],
                [-6.0295], [-5.9937], [-5.9651], [-5.8888], [-5.8137], [-5.7405], [-5.7429], [-5.8212], [-5.8967], [-5.9552], [-5.9658], [-5.9283],
                [-5.9219], [-5.9360], [-5.9943], [-6.0838], [-6.1482], [-6.2169], [-6.2732], [-6.3252], [-6.4438], [-6.6830], [-6.9697], [-7.1962],
                [-7.3519], [-7.3759], [-7.3302], [-7.1762], [-6.9551], [-6.7458], [-6.6292], [-6.5967]]).float()
        self._std = torch.tensor([[0.9304], [0.7729], [1.0068], [1.5478], [1.8270], [1.7940], [1.6933], [1.7043], [1.8344], [1.8844], [1.8506], [1.7672], [1.7807],
                [1.7977], [1.7882], [1.7599], [1.7680], [1.7909], [1.7831], [1.7588], [1.7445], [1.7822], [1.7940], [1.7761], [1.7961], [1.7989],
                [1.7818], [1.7519], [1.7466], [1.7335], [1.7068], [1.7336], [1.7537], [1.7538], [1.7427], [1.7253], [1.7055], [1.7193], [1.7359],
                [1.7460], [1.7527], [1.7514], [1.7380], [1.7031], [1.6757], [1.6612], [1.6603], [1.6675], [1.7022], [1.7513], [1.7748], [1.7932],
                [1.7957], [1.8250], [1.8481], [1.8137], [1.7564], [1.7130], [1.7024], [1.7243], [1.7348], [1.7485], [1.7810], [1.8169], [1.8318],
                [1.8312], [1.8427], [1.8756], [1.9143], [1.9503], [2.0072], [2.0761], [2.1519], [2.1848], [2.1574], [2.1386], [2.1442], [2.1601],
                [2.1547], [2.1208]]).float()

        self.len = start_len-step
        self._step = step

    def step(self):
        self.len += self._step
        if self.len >= 900: self.len = 900
        print("data len ", self.len)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
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
