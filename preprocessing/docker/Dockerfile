# minimal bioconductor image for deploying flowSOM
FROM bioconductor/release_core2

# RUN apt-get update && \
# 	apt-get -y install libxml2-dev libcurl4-openssl-dev libgeos-dev libssl-dev git

ADD docker/install_base.R /tmp/
RUN R -f "/tmp/install_base.R"

RUN git clone https://github.com/xiamaz/flowProc.git /tmp

ADD docker/install_extra.R /tmp/
RUN R -f "/tmp/install_extra.R"

# adding user repositories
ADD preprocessing /app
WORKDIR "/app/"

ENTRYPOINT ["make", "run"]
