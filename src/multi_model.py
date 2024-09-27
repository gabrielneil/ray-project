"""
@serve.deployment(name="Translator_model",
                  num_replicas=2,
                  ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
                  max_ongoing_requests=100,
                  health_check_period_s=10,
                  health_check_timeout_s=30,
                  graceful_shutdown_timeout_s=20,
                  graceful_shutdown_wait_loop_s=2)
class Translator:
    def __init__(self):
        self.language = "french"
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        model_output = self.model(text)

        translation = model_output[0]["translation_text"]

        return translation

    def reconfigure(self, config: Dict):
        self.language = config.get("language", "french")

        if self.language.lower() == "french":
            self.model = pipeline("translation_en_to_fr", model="t5-small")
        elif self.language.lower() == "german":
            self.model = pipeline("translation_en_to_de", model="t5-small")
        elif self.language.lower() == "romanian":
            self.model = pipeline("translation_en_to_ro", model="t5-small")
        else:
            pass
"""

"""
@serve.deployment(name="Summarizer_model",
                  num_replicas=2,
                  ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
                  max_ongoing_requests=100,
                  health_check_period_s=10,
                  health_check_timeout_s=30,
                  graceful_shutdown_timeout_s=20,
                  graceful_shutdown_wait_loop_s=2)
class Summarizer:
    def __init__(self, translator: DeploymentHandle):
        # Load model
        self.model = pipeline("summarization", model="t5-small")
        self.translator = translator
        self.min_length = 5
        self.max_length = 15

    def summarize(self, text: str) -> str:
        # Run inference
        model_output = self.model(
            text, min_length=self.min_length, max_length=self.max_length
        )

        # Post-process output to return only the summary text
        summary = model_output[0]["summary_text"]

        return summary

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        summary = self.summarize(english_text)

        return await self.translator.translate.remote(summary)

    def reconfigure(self, config: Dict):
        self.min_length = config.get("min_length", 5)
        self.max_length = config.get("max_length", 15)


app = Summarizer.bind(Translator.bind())
"""

from io import BytesIO

import requests
import starlette
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import pipeline


@serve.deployment
def downloader(image_url: str):
    image_bytes = requests.get(image_url).content
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


@serve.deployment(name="ImageClassifier",
                  num_replicas=2,
                  ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
                  max_ongoing_requests=100,
                  health_check_period_s=10,
                  health_check_timeout_s=30,
                  graceful_shutdown_timeout_s=20,
                  graceful_shutdown_wait_loop_s=2)
class ImageClassifier:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        self.model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    async def classify(self, image_url: str) -> str:
        image = await self.downloader.remote(image_url)
        results = self.model(image)
        return results

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return await self.classify(req["image_url"])


app = ImageClassifier.options(route_prefix="/classify").bind(downloader.bind())



