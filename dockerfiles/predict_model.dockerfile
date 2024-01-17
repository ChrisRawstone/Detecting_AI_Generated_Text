FROM --platform=linux/amd64 python:3.11-slim 

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install evidently
RUN pip install python-multipart

COPY src/ src/
COPY predict_api.py predict_api.py
COPY pyproject.toml pyproject.toml 

RUN pip install . --no-cache-dir --no-deps

EXPOSE 8080

CMD exec uvicorn predict_api:app --port $PORT --host 0.0.0.0 --workers 1