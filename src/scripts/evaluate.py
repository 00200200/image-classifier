import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.data.dataio import ImageDataModule
from src.model.model import Resnet50


def evaluate():
    model = Resnet50.load_from_checkpoint(
        "lightning_logs/version_0/checkpoints/best_model-epoch=13-val_loss=0.01.ckpt"
    )
    data = ImageDataModule()
    data.setup("test")

    trainer = L.Trainer(accelerator="mps")
    predictions = trainer.predict(model, data.test_dataloader())

    all_preds = torch.cat([batch["preds"] for batch in predictions])
    all_labels = torch.cat([batch["targets"] for batch in predictions])

    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    class_names = data.test_dataset.classes

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("assets/docs/confusion_matrix.png")
    plt.show()

    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    evaluate()
