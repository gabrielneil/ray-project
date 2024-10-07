from io import BytesIO
from typing import Dict

import ray
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from ray import serve
from torchvision import transforms
from torchvision.models import resnet18
from transformers import pipeline

app = FastAPI()
prediction_endpoint = "/predict"
ray.init(address="auto")
serve.start(detached=True)


# 1: Create an image classifier model with Pytorch and serve it with Ray + FastAPI.
@serve.deployment
@serve.ingress(app)
class ImgClassification:
    def __init__(self):
        self.count = 0
        self.model = resnet18(pretrained=True).eval()
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda t: t[:3, ...]
                ),  # remove the alpha channel
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def classify(self, image_payload_bytes):
        pil_image = Image.open(BytesIO(image_payload_bytes))

        pil_images = [pil_image]  # batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images]
        )

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        return {"class_index": int(torch.argmax(output_tensor[0]))}

    @app.post(prediction_endpoint)
    async def prediction(self, file: UploadFile = File(...)):
        image_bytes = await file.read()
        return self.classify(image_bytes)


# 2: Wrap the HuggingFace's pretrained sentiment analysis model in a Serve deployment with Ray + FastAPI.
# 2.1: Define request model (pydantic)
class SentimentAnalysisRequest(BaseModel):
    text: str


# 2.2: Create model and endpoint
@serve.deployment(num_replicas=2, max_concurrent_queries=15)
@serve.ingress(app)
class SentimentAnalysis:
    def __init__(self):
        self._model = pipeline("sentiment-analysis")

    @app.post(prediction_endpoint)
    def prediction(self, sentiment_request: SentimentAnalysisRequest) -> Dict:
        return self._model(sentiment_request.text)[0]


# 3: Deployment of both models
ImgClassification.deploy()
SentimentAnalysis.deploy()
