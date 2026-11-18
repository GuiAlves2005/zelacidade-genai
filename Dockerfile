# Imagem da API da Zela (CPU). Para GPU, troque a base por uma imagem CUDA do PyTorch.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/hf
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY base/ base/
COPY avaliacao/limiar.json avaliacao/limiar.json
COPY modelos/ modelos/

RUN useradd -m zela && chown -R zela /app
USER zela

EXPOSE 8001
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8001/saude')" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
