FROM --platform=linux/amd64 python:3.11-slim 

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

# Install dependencies (this will be cached if requirements.txt doesn't change)

RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
#RUN pip install -r requirements.txt





