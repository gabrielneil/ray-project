import logging
from io import BytesIO
from urllib.request import Request

import requests
import scipy
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from moviepy.editor import *
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import pipeline

# Set up logging

logger = logging.getLogger("ray.serve")


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
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    health_check_period_s=10,
    health_check_timeout_s=500,
    graceful_shutdown_timeout_s=500,
    graceful_shutdown_wait_loop_s=2,
)
class ImageClassifier:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        self.model = pipeline(
            "image-to-text", model="Salesforce/blip-image-captioning-base"
        )

    async def classify(self, image_url: str) -> dict:
        image_ref = self.downloader.remote(image_url)
        image = await image_ref

        logger.info("Processing image in ImageClassifier.")

        caption = self.model(image)[0]["generated_text"]
        return {"response": caption}


# SentimentAnalysis Deployment
@serve.deployment(
    name="SentimentAnalysis",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    health_check_period_s=10,
    health_check_timeout_s=500,
    graceful_shutdown_timeout_s=500,
    graceful_shutdown_wait_loop_s=2,
)
class SentimentAnalysis:
    def __init__(self):
        self.model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    async def analyze_sentiment(self, text: str) -> dict:
        # Run inference on the text
        model_output = self.model(text)[0]
        logger.info("Performed sentiment analysis.")
        return model_output


# CoverAlbumMaker Deployment
@serve.deployment(
    name="CoverAlbumMaker",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    health_check_period_s=10,
    health_check_timeout_s=500,
    graceful_shutdown_timeout_s=500,
    graceful_shutdown_wait_loop_s=2,
)
class CoverAlbumMaker:
    def __init__(self):
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("mps")

    async def make_cover(self, text) -> dict:
        # Run inference on the text
        prompt = f"Album cover: {text}"
        image = self.pipe(prompt).images[0]
        image.save("ray_cover_album_output.png")
        return {"result": "ok"}


# VideoMaker Deployment
@serve.deployment(
    name="VideoMaker",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    health_check_period_s=10,
    health_check_timeout_s=500,
    graceful_shutdown_timeout_s=500,
    graceful_shutdown_wait_loop_s=2,
)
class VideoMaker:
    def __init__(self):
        # Define the paths to your image and audio files
        self.image_path = (
            "ray_cover_album_output.png"  # Path to your PNG image
        )
        self.audio_path = "ray_musicgen_out.wav"  # Path to your WAV audio
        self.output_video_path = (
            "ray_final_video.mp4"  # Path to save the output video
        )

    async def make_video(self) -> dict:
        # Run inference on the text
        logger.info("Performed sentiment analysis.")
        # Load the audio file
        audio_clip = AudioFileClip(self.audio_path)

        # Load the image and set the duration to match the audio file's duration
        image_clip = ImageClip(self.image_path).set_duration(
            audio_clip.duration
        )

        # Explicitly set the frame rate (FPS) for the video
        fps = 24  # Use a common frame rate
        image_clip = image_clip.set_fps(fps)

        # Print the fps to debug
        print(f"FPS value: {fps}")

        # Check if fps is set correctly
        if fps is None:
            raise ValueError("FPS value is None. Please set a valid FPS.")

        # Set the audio to the image clip
        video_clip = image_clip.set_audio(audio_clip)

        # Write the video file
        video_clip.write_videofile(
            self.output_video_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,  # Pass the fps explicitly
        )

        return {"ok": "model_output"}


# MusicMaker Deployment
@serve.deployment(
    name="MusicMaker",
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 0},
    health_check_period_s=10,
    health_check_timeout_s=500,
    graceful_shutdown_timeout_s=500,
    graceful_shutdown_wait_loop_s=2,
)
class MusicMaker:
    def __init__(self):
        self.model = pipeline(
            "text-to-audio", model="facebook/musicgen-small"  # , device=device
        )

    async def process_music(self, text: str) -> dict:
        # Run inference on the text
        logger.info("Processing music!")
        music = self.model(text, forward_params={"do_sample": True})

        logger.info("It's HERE!!!!!")

        scipy.io.wavfile.write(
            "ray_musicgen_out.wav",
            rate=music["sampling_rate"],
            data=music["audio"],
        )

        return {"ok": "model_output"}


# WrapperModels Deployment
@serve.deployment(
    name="WrapperModels",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    health_check_period_s=10,
    health_check_timeout_s=500,
    graceful_shutdown_timeout_s=500,
    graceful_shutdown_wait_loop_s=2,
)
class WrapperModels:
    def __init__(
        self,
        image_classifier: DeploymentHandle,
        sentiment_analysis: DeploymentHandle,
        cover_album_maker: DeploymentHandle,
        music_maker: DeploymentHandle,
        video_maker: DeploymentHandle,
    ):
        # Load sentiment analysis model
        self.model = pipeline("text-to-audio", model="facebook/musicgen-small")
        self.image_classifier = image_classifier
        self.sentiment_analysis = sentiment_analysis
        self.cover_album_maker = cover_album_maker
        self.music_maker = music_maker
        self.video_maker = video_maker

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
        # logger.info("Received request in SentimentAnalysis.", image_url)

        # Get the caption from the ImageClassifier
        caption_result_ref = self.image_classifier.classify.remote(image_url)
        caption_result = await caption_result_ref  # Await the ObjectRef

        caption = caption_result["response"]

        logger.info("Caption obtained from ImageClassifier: %s", caption)

        # Perform sentiment analysis on the caption
        sentiment = self.sentiment_analysis.analyze_sentiment.remote(caption)
        sentiment_result = await sentiment  # Await the ObjectRef

        logger.info(f"SENTIMENT ANALISYS RESULTTTTT: {sentiment_result}")

        # Step 3: Generate music based on the sentiment and caption
        music_result = await self.generate_music(
            sentiment_result["label"], caption
        )
        logger.info(f"Generated music result!!!: {music_result}")

        cover = self.cover_album_maker.make_cover.remote(caption)
        cover_result = await cover  # Await the ObjectRef

        logger.info(f"COVER RESULTTTTT!!!: {cover_result}")

        process_music = self.music_maker.process_music.remote(caption)
        process_music_result = await process_music

        logger.info(f"PROCESSS MUSSICCCC!!!: {process_music_result}")
        # process_music = "jaja"
        logger.info(f"Generated music process result: {process_music_result}")

        video = self.video_maker.make_video.remote()
        video_result = await video  # Await the ObjectRef

        logger.info(f"Generated VIDEO!!!!: {video_result}")
        # Return both the caption and the sentiment analysis result
        return {
            "caption": caption,
            "sentiment": sentiment_result,
            "music": music_result,
        }


# Bind deployments
app = WrapperModels.options(route_prefix="/classify").bind(
    ImageClassifier.bind(Downloader.bind()),
    SentimentAnalysis.bind(),
    CoverAlbumMaker.bind(),
    MusicMaker.bind(),
    VideoMaker.bind(),
)
