# minimal bioconductor image for deploying flowSOM
FROM python:3.6

RUN pip install awscli && apt-get update && apt-get install git

# adding user repositories
ADD classification /app
WORKDIR "/app/"

RUN pip install -r requirements.txt

ENTRYPOINT ["make", "run"]
