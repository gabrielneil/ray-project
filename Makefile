deployment_port=8000
sentiment_endpoint_name=sentiment
text_to_predict=Ray+Serve+is+great%21
install:
	@( \
		python3.8 -m venv venv; \
		source venv/bin/activate; \
		python -m pip install --no-cache-dir -r requirements.txt; \
	)

start-and-serve:
	@( \
		ray start --head; \
		serve start --http-host=0.0.0.0; \
	)

stop:
	@( \
		ray stop; \
	)

deploy-single-model:
	@( \
		python src/single_model.py; \
	)

deploy-multiple-models:
	@( \
		python src/multiple_models.py; \
	)

single-model-predictions:
	@( \
		curl -X GET \
		-H 'Content-Type: application/json' \
		http://localhost:$(deployment_port)/$(sentiment_endpoint_name)?text=$(text_to_predict); \
	)

multiple-models-predictions:
	@( \
        curl --request POST \
          http://localhost:$(deployment_port)/SentimentAnalysis/predict \
          -H 'Content-Type: application/json' \
          -d '{"text": "Ray serve is great"}'; \
	)