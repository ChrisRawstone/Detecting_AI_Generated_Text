FROM python:3.11-slim 

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir


RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install evidently



COPY src/ src/
COPY predict_api.py predict_api.py
COPY pyproject.toml pyproject.toml 

RUN pip install . --no-cache-dir --no-deps

ENTRYPOINT ["uvicorn", "predict_api:app", "--port","8020", "--host", "0.0.0.0", "--workers", "1"]