FROM exel232/tensorflow-conda:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY flowcat /flowcat
