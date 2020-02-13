import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir, wandb):
        # logdir = logdir / current time _ hostname
        import socket
        import os
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join(logdir, current_time +
                              '_' + socket.gethostname())
        self.wandb = wandb

        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

        self.wandb.log({"training loss": reduced_loss, "grad norm": grad_norm,
                        "learning rate": learning_rate, "duration": duration, })

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = 0  # random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        if gate_outputs is not None:
            self.add_image(
                "gate",
                plot_gate_outputs_to_numpy(
                    gate_targets[idx].data.cpu().numpy(),
                    torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
                iteration, dataformats='HWC')

        align = plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T)
        spec = plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy())
        mel = plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy())
        gate = plot_gate_outputs_to_numpy(gate_targets[idx].data.cpu(
        ).numpy(), torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())

        wandb = self.wandb
        wandb.log({"validation loss": reduced_loss, "alignment": wandb.Image(
            align), "spectrogram": wandb.Image(spec), "mel_spec": wandb.Image(mel), "gate": wandb.Image(gate), })
