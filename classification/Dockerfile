# minimal bioconductor image for deploying flowSOM
FROM python:3.6

RUN pip install awscli && apt-get update && apt-get install -y git fish

# adding user repositories
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app
WORKDIR "/app/"

ENTRYPOINT ["./run_s3.fish"]
