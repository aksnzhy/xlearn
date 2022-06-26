FROM python:3.7-slim-buster as base

RUN apt update && \
    apt install -y cmake g++ && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /xlearn

# build from source (installs to /usr/local/lib/python3.7/site-packages/)
RUN cd /xlearn && \
    ls -l && \
    ./build.sh && \
    rm -r /xlearn

# install other library (installs to same site-packages path)
RUN pip install --no-cache-dir liblinear==2.11.2

# create an image without build dependencies
FROM python:3.7-slim-buster AS lib
ENV USER=root
COPY --from=base /usr/local/lib/python3.7/site-packages/* /usr/local/lib/python3.7/
