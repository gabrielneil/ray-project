from io import BytesIO
from typing import Dict

import ray
import torch
from PIL import Image
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from torchvision import transforms
from torchvision.models import resnet18
from transformers import pipeline

app = FastAPI()
prediction_endpoint = "prediction"
ray.init(address="auto")
serve.start(detached=True)


@serve.deployment
@serve.ingress(app)
class ModelServer:
    def __init__(self):
        self.count = 0
        self.model = resnet18(pretrained=True).eval()
        self.preprocessor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t[:3, ...]),  # remove the alpha channel
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, image_payload_bytes):
        pil_image = Image.open(BytesIO(image_payload_bytes))

        pil_images = [pil_image]  # batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images])

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        return {"class_index": int(torch.argmax(output_tensor[0]))}

    @app.post("/prediction_endpoint")
    def prediction(self):
        return "Welcome to the PyTorch model server."


@serve.deployment
@serve.ingress(app)
class SentimentAnalysisServer:
    def __init__(self):
        self._model = pipeline("sentiment-analysis")

    @app.post("/predict")
    def get(self, request: Request) -> Dict:
        return self._model(request.query_params["text"])[0]


ModelServer.deploy()
SentimentAnalysisServer.deploy()
