# read from the base image
FROM gcr.io/elite-totem-410916/base_image:latest


COPY src/ src/
COPY predict_api.py predict_api.py
COPY pyproject.toml pyproject.toml 

RUN pip install . --no-cache-dir --no-deps

# ENTRYPOINT ["uvicorn", "predict_api:app", "--port","8080", "--host", "0.0.0.0", "--workers", "1"]

EXPOSE 8080

CMD exec uvicorn predict_api:app --port $PORT --host 0.0.0.0 --workers 1
# CMD ["uvicorn", "predict_api:app", "--port", "8080", "--host", "0.0.0.0", "--workers", "1"]