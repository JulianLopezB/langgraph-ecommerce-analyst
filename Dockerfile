FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The application fetches secrets from AWS Secrets Manager at runtime.
# Provide GEMINI_SECRET_NAME, LANGCHAIN_SECRET_NAME and AWS_REGION as environment variables.
CMD ["python", "main.py"]
