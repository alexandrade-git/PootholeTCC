# Camera Receiver (Detecção de Buracos)
````markdown
# Camera Receiver (Detecção de Buracos)

Projeto FastAPI que recebe imagens de câmeras, executa inferência (YOLO / Ultralytics), salva imagens recebidas e anotadas, e expõe um dashboard web.

## Estrutura principal

- `app.py` - API FastAPI e lógica de inferência
- `send_from_camera.py` - cliente que envia frames (motion-detection)
- `dashboard.html` - dashboard frontend (frontend estático)
- `requirements.txt` - dependências Python
- `best.pt` - modelo treinado 

## Requisitos

- Python 3.8+ (recomendado criar um virtualenv)
- `pip` instalado
- (Opcional) Docker e Docker Compose

## Preparar ambiente (PowerShell)

1. Criar e ativar virtualenv (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependências:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Variáveis de ambiente (exemplo PowerShell):

```powershell
#$Env:API_KEY = "ak_..."
#$Env:AZURE_CONN_STR = "DefaultEndpointsProtocol=..."
#$Env:MODEL_PATH = "D:\\caminho\\para\\best.pt"
```

> Observação: não coloque chaves/segredos diretamente no repositório. Use GitHub Secrets para CI/CD.

## Rodando localmente (sem Docker)

Se a aplicação usa `uvicorn` (FastAPI), execute:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Abra `http://localhost:8000` (ou a rota que sua `app.py` expõe). O `dashboard.html` pode ser acessado em `/dashboard.html` ou abrindo o arquivo localmente, dependendo de como o backend serve arquivos estáticos.

Para enviar frames de uma câmera/local (cliente):

```powershell
python send_from_camera.py
```

## Rodando com Docker Compose

1. Ajuste o `Dockerfile` / `docker-compose.yml` se necessário.
2. Build e subir:

```powershell
docker compose build
docker compose up -d
```

A aplicação ficará exposta em `http://localhost:8000`.
