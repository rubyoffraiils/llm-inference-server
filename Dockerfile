# CPU image. The GPU benchmark runs on a machine with CUDA drivers and
# its own PyTorch build; this is the one you deploy.
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch. The default wheel pulls ~2GB of CUDA libraries that do
# nothing without a GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

COPY src/ src/
COPY static/ static/

# Weights land here, so mounting a volume at this path survives restarts
# and keeps the image small.
ENV HF_HOME=/cache
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Both models download on first boot, which takes a minute -- the health
# check gives it room before reporting unhealthy.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/stats')"

WORKDIR /app/src
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
