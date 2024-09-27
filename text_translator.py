import starlette
from ray import serve
from transformers import pipeline


@serve.deployment(name="Translator_model!!!!",
                  num_replicas=2,
                  ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
                  max_ongoing_requests=100,
                  health_check_period_s=10,
                  health_check_timeout_s=30,
                  graceful_shutdown_timeout_s=20,
                  graceful_shutdown_wait_loop_s=2)
class Translator:
    def __init__(self):
        self.model = pipeline("translation_en_to_de", model="t5-small")

    def translate(self, text: str) -> str:
        return self.model(text)[0]["translation_text"]

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return self.translate(req["text"])


app = Translator.options(route_prefix="/translate").bind()
