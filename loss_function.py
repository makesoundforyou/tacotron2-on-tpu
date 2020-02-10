from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, maxlen):
        super(Tacotron2Loss, self).__init__()
        self._maxlen = maxlen

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss(reduction='sum')(mel_out, mel_target) + \
            nn.MSELoss(reduction='sum')(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss(reduction='sum')(gate_out, gate_target)
        mel_loss = mel_loss / self._maxlen / mel_out.shape[0] / mel_out.shape[1]
        gate_loss = gate_loss  / self._maxlen / mel_out.shape[0]
        return mel_loss + gate_loss
