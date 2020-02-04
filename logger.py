import random
import torch
# from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger():
    def __init__(self, wandb):
        self.wandb = wandb

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            # self.add_scalar("training.loss", reduced_loss, iteration)
            # self.add_scalar("grad.norm", grad_norm, iteration)
            # self.add_scalar("learning.rate", learning_rate, iteration)
            # self.add_scalar("duration", duration, iteration)
            self.wandb.log({
                "training loss": reduced_loss,
                "grad norm": grad_norm,
                "learning rate": learning_rate,
                "duration": duration,
            })

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        # self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        align = plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T)
        spec = plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy())
        mel = plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy())
        gate = plot_gate_outputs_to_numpy( gate_targets[idx].data.cpu().numpy(), torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())

        wandb = self.wandb
        wandb.log({
            "validation loss": reduced_loss,
            "alignment": wandb.Image(align),
            "spectrogram": wandb.Image(spec),
            "mel_spec": wandb.Image(mel),
            "gate": wandb.Image(gate),
            })

        # self.add_image( "alignment", align, iteration)
        # self.add_image( "mel_target", spec, iteration)
        # self.add_image( "mel_predicted", mel, iteration)
        # self.add_image( "gate", gate, iteration)
