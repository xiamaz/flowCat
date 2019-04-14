TAG = exel232/flowcat:latest

build:
	docker build -t $(TAG) .

push:
	docker push $(TAG)

run:
	docker run -it $(TAG) /bin/bash
