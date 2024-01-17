# Base image
FROM base_image:latest


COPY pyproject.toml pyproject.toml
COPY Makefile Makefile

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
RUN dvc config core.no_scm true
COPY data.dvc data.dvc

COPY src/ src/

WORKDIR /

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["sh","-c","dvc pull data/processed/tokenized_data/full_data && python src/train_model.py experiment=experiment_1"]