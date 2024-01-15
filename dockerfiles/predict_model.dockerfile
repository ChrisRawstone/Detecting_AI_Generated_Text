# Base image
FROM python:3.11-slim


RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN pip install -r requirements.txt --no-cache-dir

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
RUN dvc config core.no_scm true
COPY data.dvc data.dvc

COPY src/ src/

RUN dvc pull

EXPOSE $PORT



RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install evidently

COPY iris_fastapi.py iris_fastapi.py


ENTRYPOINT ["uvicorn", "iris_fastapi:app", "--port","8020", "--host", "0.0.0.0", "--workers", "1"]
#CMD exec uvicorn iris_fastapi:app --port $PORT --host 0.0.0.0 --workers 1