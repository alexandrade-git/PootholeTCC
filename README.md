# Camera Receiver (Detecção de Buracos)
````markdown
# Camera Receiver (Detecção de Buracos)

Projeto FastAPI que recebe imagens de câmeras, executa inferência (YOLO / Ultralytics), salva imagens recebidas e anotadas, e expõe um dashboard web simples.

## Estrutura principal

- `app.py` - API FastAPI e lógica de inferência
- `send_from_camera.py` - cliente que envia frames (motion-detection)
- `dashboard.html` - dashboard frontend (frontend estático)
- `requirements.txt` - dependências Python
- `best.pt` - modelo treinado (não recomendado commitar no Git diretamente)

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

## Como subir este projeto no GitHub (passo a passo)

Estas instruções são para PowerShell no Windows.

1) Inicializar o repositório local (se ainda não for um repo):

```powershell
cd "D:\\Project Codes\\UNIP TCC"
git init
git add .
git commit -m "Initial commit"
```

2) Criar branch principal (opcional, força `main`):

```powershell
git branch -M main
```

3) Criar repositório remoto no GitHub e enviar:

- Opção A — Usando a interface web: crie um repo no GitHub e copie a URL HTTPS.

```powershell
git remote add origin https://github.com/<SEU_USUARIO>/<NOME_REPO>.git
git push -u origin main
```

- Opção B — Usando GitHub CLI (`gh`) (recomendado se você tiver o `gh` instalado e autenticado):

```powershell
gh repo create <SEU_USUARIO>/<NOME_REPO> --public --source . --remote origin --push
```

4) Arquivos grandes / modelos (`best.pt`)

O GitHub não aceita arquivos maiores que 100 MB no push. Para modelos grandes considere:

- Não commitar o arquivo `best.pt` no repositório. Envie o modelo para um storage (Azure Blob, S3, Google Drive) e documente o download.
- Ou usar Git LFS para rastrear arquivos grandes:

```powershell
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add best.pt
git commit -m "Add model with Git LFS"
git push origin main
```

> Se o push falhar com erro sobre tamanho >100MB, remova o arquivo do histórico ou use LFS.

5) Não commitar segredos

Adicione um `.gitignore` com itens como:

```
# Python
__pycache__/
.venv/
*.pyc
# Modelos
*.pt
# Ambiente
.env
```

## Deploy / CI

Se você já tem um workflow no `.github/workflows/` ele pode buildar imagens e publicar no GitHub Packages / Container Registry. Use GitHub Secrets para credenciais (SSH, tokens, providers de nuvem).

## Troubleshooting rápido

- Erro ao rodar `uvicorn`: verifique se as dependências estão instaladas e se o virtualenv está ativado.
- Erro de push: verifique se o remote está correto (`git remote -v`) e se existe autenticação (token/SSH key).

## Contato / próxima etapa

Se quiser, eu posso:

- Rodar os comandos Git localmente aqui (se autorizar) para inicializar e fazer o primeiro push.
- Adicionar um `.gitignore` e configurar Git LFS automaticamente.

````

