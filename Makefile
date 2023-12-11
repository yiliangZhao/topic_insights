PROJECT_ID=openspace-dev

build_image:
	@docker build -t gcr.io/${PROJECT_ID}/topicinsights . -f Dockerfile

.PHONY: push_image
push_image:
	@docker push gcr.io/${PROJECT_ID}/topicinsights