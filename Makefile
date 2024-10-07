deployment_port=8000
sentiment_endpoint_name=sentiment
text_to_predict=Ray+Serve+is+great%21
install:
	@( \
		python3.10 -m venv venv; \
		source venv/bin/activate; \
		python -m pip install --no-cache-dir -r requirements.txt; \
	)

start-and-serve:
	@( \
		ray start --head; \
		serve start --http-host=0.0.0.0; \
	)

start:
	@( \
		ray start --head; \
	)

stop:
	@( \
		ray stop; \
	)

build:
	@( \
		serve build src.multi_model:app -o k8s/serve_config.yaml; \
	)

deploy:
	@( \
		serve deploy k8s/serve_config.yaml; \
	)

launch-prometheus:
	@( \
		ray metrics launch-prometheus; \
	)

execute-project: start build deploy

restart: stop execute-project

deploy-multiple-models:
	@( \
		python src/multiple_models.py; \
	)

multiple-models-predictions:
	@( \
        curl --request POST \
          http://localhost:$(deployment_port)/SentimentAnalysis/predict \
          -H 'Content-Type: application/json' \
          -d '{"text": "Ray serve is great"}'; \
	)

predictions:
	@( \
        curl --request POST \
          http://localhost:$(deployment_port)/classify \
          -H 'Content-Type: application/json' \
          -d '{"image_url": "https://www.iptmiami.com/static/sitefiles/images/ipt-joy-01.jpg"}'; \
	)