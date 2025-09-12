FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies for Prophet/cmdstanpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and pre-build CmdStan
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m cmdstanpy.install_cmdstan --yes

# Copy project files
COPY . .

# Default command
CMD ["python", "main.py"]
