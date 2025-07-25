FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY upload_model_app.py .
EXPOSE 8001
CMD ["uvicorn", "upload_model_app:app", "--host", "0.0.0.0", "--port", "8001"]
