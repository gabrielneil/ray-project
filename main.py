from typing import Dict

from ray import serve
from starlette.requests import Request
from transformers import pipeline


# 1: Wrap the pretrained sentiment analysis model in a Serve deployment.
@serve.deployment(route_prefix="/sentiment")
class SentimentAnalysisDeployment:
    def __init__(self):
        self._model = pipeline("sentiment-analysis")

    def __call__(self, request: Request) -> Dict:
        return self._model(request.query_params["text"])[0]


# 2: Deploy the deployment.
serve.run(SentimentAnalysisDeployment.bind())

# 3: Query the deployment and print the result.
#print(requests.get("http://localhost:8000/", params={"text": "Ray Serve is great!"}).json())
# {'label': 'POSITIVE', 'score': 0.9998476505279541}
