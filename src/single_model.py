from typing import Dict

from ray import serve
from starlette.requests import Request
from transformers import pipeline


# 1: Wrap the HuggingFace's pretrained sentiment analysis model in a Serve deployment.
@serve.deployment(route_prefix="/sentiment")
class SentimentAnalysisDeployment:
    def __init__(self):
        self._model = pipeline("sentiment-analysis")

    def __call__(self, request: Request) -> Dict:
        return self._model(request.query_params["text"])[0]


# 2: Deployment of the model.
serve.run(SentimentAnalysisDeployment.bind())
