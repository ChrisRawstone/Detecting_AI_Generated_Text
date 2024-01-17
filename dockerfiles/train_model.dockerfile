# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY Makefile Makefile

RUN pip install -r requirements.txt --no-cache-dir

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
RUN dvc config core.no_scm true
COPY data.dvc data.dvc

COPY src/ src/

WORKDIR /

RUN pip install -e .

ENTRYPOINT ["sh","-c","dvc pull data/processed/tokenized_data && python src/train_model.py experiment=experiment_1"]