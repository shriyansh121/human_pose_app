# ✅ 1. Base Image (Python 3.10 is perfect for MediaPipe + OpenCV)
FROM python:3.10-slim

# ✅ 2. Set Working Directory inside container
WORKDIR /app

# ✅ 3. Install System Dependencies (important for OpenCV & MediaPipe)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ✅ 4. Copy only requirements first (for faster rebuilds)
COPY requirements.txt .

# ✅ 5. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 6. Copy Full Project into Container
COPY . .

# ✅ 7. Expose Streamlit Port
EXPOSE 8501

# ✅ 8. Environment Config (Prevents Streamlit Warnings)
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ✅ 9. Run Streamlit App
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
