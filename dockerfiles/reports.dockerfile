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

COPY src/ src/ 
COPY reports_api.py reports_api.py
COPY pyproject.toml pyproject.toml 
COPY reports/ reports/

RUN pip install . --no-cache-dir --no-deps

EXPOSE 8080

CMD exec uvicorn reports_api:app --port $PORT --host 0.0.0.0 --workers 1