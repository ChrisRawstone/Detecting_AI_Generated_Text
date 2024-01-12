# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml


RUN pip install -r requirements.txt --no-cache-dir

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
RUN dvc config core.no_scm true
COPY data.dvc data.dvc

RUN dvc pull --verbose

COPY src/ src/

WORKDIR /

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/train_model.py"]