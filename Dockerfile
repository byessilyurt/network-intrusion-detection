# Production Dockerfile for Network Intrusion Detection System
# Includes FastAPI REST API + Streamlit Dashboard
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir 'urllib3<2.0' streamlit plotly

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/

# Create data directory (for optional CICIDS2017 samples)
RUN mkdir -p data/raw

# Create non-root user for security
RUN useradd -m -u 1000 nids && \
    chown -R nids:nids /app

USER nids

# Expose ports
EXPOSE 8000 8501

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start both API and Dashboard
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
