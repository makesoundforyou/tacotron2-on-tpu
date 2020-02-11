from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, len):
        super(Tacotron2Loss, self).__init__()
        self.len = len

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        mel_loss = nn.MSELoss(reduction='sum')(mel_out, mel_target) + \
            nn.MSELoss(reduction='sum')(mel_out_postnet, mel_target)
        mel_loss /= mel_out.shape[0] * mel_out.shape[1] * self.len

        if gate_out is None: return mel_loss

        gate_out = gate_out.view(-1, 1)
        gate_loss = nn.BCEWithLogitsLoss(reduction='sum')(gate_out, gate_target)
        gate_loss /= mel_out.shape[0] * self.len
        return mel_loss + gate_loss
