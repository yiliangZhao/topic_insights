build_image:
	@docker build -t gcr.io/${PROJECT_ID}/${IMAGE_NAME} . -f Dockerfile

.PHONY: push_image
push_image:
	@docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}