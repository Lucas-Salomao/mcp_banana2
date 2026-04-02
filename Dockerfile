FROM python:3.12-slim

# Evita arquivos .pyc e permite logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Instala dependências do sistema necessárias para Pillow/GCS (se houver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do servidor
COPY banana_mcp_sse.py .

# O GKE gerencia as credenciais via Workload Identity (KSA -> GSA), 
# então não copiamos o service_account.json para a imagem por segurança.

EXPOSE 8080

# Inicia o servidor SSE
CMD ["python", "banana_mcp_sse.py"]
