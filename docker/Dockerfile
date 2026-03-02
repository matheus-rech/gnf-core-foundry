# gnf-core-foundry Docker image
# Python 3.11 + R 4.x + irr package + all Python dependencies

FROM python:3.11-slim-bookworm

LABEL maintainer="Global NeuroFoundry <engineering@globalneuro.org>"
LABEL description="gnf-core-foundry — Digital Biomarker Pipeline Engine"
LABEL version="0.1.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    pkg-config \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libopenblas-dev \
    liblapack-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e "install.packages('irr', repos='https://cloud.r-project.org', quiet=TRUE)"
RUN Rscript -e "library(irr); cat('irr package installed successfully\n')"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src
ENV R_HOME=/usr/lib/R
ENV LD_LIBRARY_PATH=/usr/lib/R/lib:${LD_LIBRARY_PATH}
ENV RPY2_CFFI_MODE=ABI

EXPOSE 8000

RUN mkdir -p /app/outputs /app/experiments

CMD ["uvicorn", "api.endpoints:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
