# minimal bioconductor image for deploying flowSOM
FROM floydhub/dl-docker:cpu

# adding user repositories
ADD classification /app
WORKDIR "/app/"

RUN pip install -r requirements.txt

ENTRYPOINT ["make", "run"]
