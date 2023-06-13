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

deploy-model:
	@( \
		python main.py; \
	)

predictions:
	@( \
		curl -X GET \
		-H 'Content-Type: application/json' \
		http://localhost:$(deployment_port)/$(sentiment_endpoint_name)?text=$(text_to_predict); \
	)