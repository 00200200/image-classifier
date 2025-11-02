import gradio as gr
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.model.model import Resnet50

model = Resnet50.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/best_model-epoch=13-val_loss=0.01.ckpt"
)
import numpy as np

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

target_layers = [model.feature_extractor[-2]]
cam = GradCAM(model=model, target_layers=target_layers)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(image):
    img = transform(image).unsqueeze(0)

    img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]
        top_id = probs.argmax()

    target = [ClassifierOutputTarget(top_id)]
    grayscale_cam = cam(input_tensor=img, targets=target)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return {CLASSES[top_id]: float(probs[top_id])}, Image.fromarray(visualization)


with gr.Blocks() as demo:
    gr.Markdown("## RESNET50 + GradCam")
    input = gr.Image(type="pil")
    btn = gr.Button("Predict")
    cam_output = gr.Image(label="Grad-CAM Visualization")
    pred = gr.Label(label="Predicted Class")

    btn.click(fn=predict, inputs=input, outputs=[pred, cam_output])


demo.launch()
