import gradio as gr
import torch
from torchvision import transforms

from src.model.model import Resnet50

model = Resnet50.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/best_model-epoch=13-val_loss=0.01.ckpt"
)
model.to("cpu").eval()
CLASSES = [
    "anchor",
    "balloon",
    "bicycle",
    "envelope",
    "paper_boat",
    "peace_symbol",
    "smiley",
    "speech_bubble",
    "spiral",
    "thumb",
]


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]

    top_prob, top_id = probs.max(dim=0)
    top_class = CLASSES[top_id]

    return {top_class: float(top_prob)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
)

demo.launch()
