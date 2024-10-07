import logging
import sys
from io import BytesIO
from urllib.request import Request

import requests
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.flush = sys.stdout.flush
logger.addHandler(handler)


# Downloader Deployment
@serve.deployment
class Downloader:
    async def __call__(self, image_url: str):
        image_bytes = requests.get(image_url).content
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image


# ImageClassifier Deployment
@serve.deployment(
    name="ImageClassifier",
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    # max_concurrent_queries=100,
    health_check_period_s=10,
    health_check_timeout_s=70,
    graceful_shutdown_timeout_s=70,
    graceful_shutdown_wait_loop_s=2
)
class ImageClassifier:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        self.model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    async def classify(self, image_url: str) -> dict:
        image_ref = self.downloader.remote(image_url)
        image = await image_ref  # Await the ObjectRef

        logger.info("Processing image in ImageClassifier.")

        caption = self.model(image)[0]["generated_text"]
        return {"response": caption}


# ImageClassifier Deployment
@serve.deployment(
    name="SentimentAnalysis",
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    # max_concurrent_queries=100,
    health_check_period_s=10,
    health_check_timeout_s=70,
    graceful_shutdown_timeout_s=70,
    graceful_shutdown_wait_loop_s=2
)
class ImageClassifier:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        self.model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    async def classify(self, image_url: str) -> dict:
        image_ref = self.downloader.remote(image_url)
        image = await image_ref  # Await the ObjectRef

        logger.info("Processing image in ImageClassifier.")

        caption = self.model(image)[0]["generated_text"]
        return {"response": caption}


# SentimentAnalysis Deployment
@serve.deployment(
    name="WrapperModels",
    num_replicas=2,
    ray_actor_options={"num_cpus": 4, "num_gpus": 0},
    # max_concurrent_queries=100,
    health_check_period_s=10,
    health_check_timeout_s=70,
    graceful_shutdown_timeout_s=70,
    graceful_shutdown_wait_loop_s=2
)
class SentimentAnalysis:
    def __init__(self, image_classifier: DeploymentHandle):
        # Load sentiment analysis model
        self.model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.image_classifier = image_classifier
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        # self.music_model = pipeline("text-to-audio", model="facebook/musicgen-small")

    def analyze_sentiment(self, text: str) -> dict:
        # Run inference on the text
        model_output = self.model(text)[0]
        logger.info("Performed sentiment analysis.")
        return model_output

    def process_music(self, text: str) -> dict:
        # Run inference on the text
        """
        music = self.music_model(text, forward_params={"do_sample": True})
        print("It's HERE!!!!!")
        scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
        """
        inputs = self.processor(
            text=["a little girl with her hands up and a flower in her mouth.",
                  "90s rock song with loud guitars and heavy drums"],
            padding=True,
            return_tensors="pt",
        )

        audio_values = self.music_model.generate(**inputs, max_new_tokens=256)
        return "music"

    async def generate_music(self, sentiment: str, caption: str) -> str:
        # Placeholder music generation based on sentiment
        if sentiment == "POSITIVE":
            return f"Generated upbeat music based on a positive mood from the caption: {caption}"
        elif sentiment == "NEGATIVE":
            return f"Generated slow, sad music based on a negative mood from the caption: {caption}"
        else:
            return f"Generated neutral music based on the caption: {caption}"

    async def __call__(self, http_request: Request) -> dict:
        request_data: dict = await http_request.json()
        image_url = request_data["image_url"]
        logger.info("Received request in SentimentAnalysis.", image_url)

        # Get the caption from the ImageClassifier
        caption_result_ref = self.image_classifier.classify.remote(image_url)
        caption_result = await caption_result_ref  # Await the ObjectRef

        caption = caption_result["response"]
        logger.info("Caption obtained from ImageClassifier: %s", caption)

        # Perform sentiment analysis on the caption
        sentiment = self.analyze_sentiment(caption)

        # Step 3: Generate music based on the sentiment and caption
        music_result = await self.generate_music(sentiment["label"], caption)
        logger.info(f"Generated music result: {music_result}")

        process_music = await self.process_music(caption)
        # process_music = "jaja"
        logger.info(f"Generated music process result: {process_music}")
        # Return both the caption and the sentiment analysis result
        return {"caption": caption, "sentiment": sentiment, "music": music_result, "process_music": process_music}


# Bind deployments
app = SentimentAnalysis.options(route_prefix="/classify").bind(
    ImageClassifier.bind(Downloader.bind())
)
