# Runtime image for the source code tagged v0.2.0 in GitHub Releases.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Machine Learning for Structured Data (MLSD)" \
      org.opencontainers.image.description="Scikit-learn compatible feature extraction for heterogeneous structured data" \
      org.opencontainers.image.source="https://github.com/xxxxxwater/Machine_Learning_For_Structured_Data" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/mlsd
COPY pyproject.toml README.md LICENSE ./
COPY MLSD ./MLSD
RUN python -m pip install --no-cache-dir . \
    && python -c "import MLSD; assert MLSD.__version__ == '0.2.0'"

COPY examples/train_mixed.py ./examples/train_mixed.py
USER 65532:65532
CMD ["python", "examples/train_mixed.py"]
