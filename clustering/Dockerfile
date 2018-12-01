FROM tensorflow/tensorflow:latest-gpu-py3

COPY requirements.txt /tmp
RUN apt-get update && apt-get install -y git awscli && pip install -r /tmp/requirements.txt

COPY . /app

WORKDIR /app
ENTRYPOINT ["make"]
