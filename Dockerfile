FROM osgeo/gdal:ubuntu-small-3.6.3

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# System dependencies similar to dea-fmc base image
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      fish \
      git \
      vim \
      htop \
      wget \
      unzip \
      python3-pip \
      libpq-dev python3-dev \
    && apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Pre-install Python requirements with constraints
RUN mkdir -p /conf
COPY requirements.txt /conf/
COPY constraints.txt /conf/
RUN pip install --no-cache-dir -r /conf/requirements.txt -c /conf/constraints.txt

# Copy source code and install package
RUN mkdir -p /code
WORKDIR /code
ADD . /code

RUN echo "Installing dea-burn-severity through the Dockerfile."
RUN pip install --no-cache-dir --extra-index-url="https://packages.dea.ga.gov.au" .

RUN pip freeze && pip check

# Basic smoke test
RUN dea-burn-severity --help
