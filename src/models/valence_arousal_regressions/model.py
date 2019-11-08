import torch.nn as nn
from ...modules.maxout import Maxout


class ValenceArousalModel(nn.Module):

    def __init__(self):
        super(ValenceArousalModel, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(3, 3)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8)
                )
        self.time_dist_fc = nn.Linear(64 * 8, 16)
        self.gru = nn.Sequential(
            nn.GRU(
                input_size=16,
                hidden_size=8,
                num_layers=2
                ),
            nn.Tanh(),
        )
        self.valence_maxout = Maxout(16, 1, 60)
        self.arousal_maxout = Maxout(16, 1, 60)

    def forward(self, mel_features):

        h = self.cnn(mel_features)
        h = self.time_dist_fc(h)
        h = self.gru(h)
        arousal = self.arousal_maxout(h)
        valence = self.valence_maxout(h)
        return arousal, valence
