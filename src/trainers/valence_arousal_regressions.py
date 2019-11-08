import torch.optim as optim
import torch.nn.functional as F


class ValenceArousalRegressionsTrainer:

    def __init__(
            self,
            model,
            n_epoch,
            data_loader,
            ):
        self.model = model
        self.optim = optim.Adam(self.model.parameters(), 1e-4)
        self.n_epoch = n_epoch
        self.data_loader = data_loader

    def train():

        for epoch in range(self.n_epoch):
            for i, (mel_features,
                    arousal_labels,
                    valence_labels) in self.data_loader:
                arousal_preds, valence_preds = self.model(mel_features)
                loss = F.mse_loss(arousal_preds, arousal_labels) + F.mse_loss(valence_preds, valence_labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                print("Epoch: {} Loss: {}".format(epoch, loss))
