import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models


class Resnet50(L.LightningModule):
    def __init__(self, num_classes: int = 10, lr: float = 1e-3):
        super().__init__()
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        all_params = list(self.feature_extractor.parameters())
        for param in all_params[:-20]:
            param.requires_grad = False

        for param in all_params[-20:]:
            param.requires_grad = True
        self.classifier = nn.Linear(num_filters, num_classes)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        logits = self.classifier(representations)
        return logits

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_accuracy", acc)
        return loss

    def predict_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
