# 🛣️ Detecção Automatizada de Buracos em Vias Urbanas usando YOLOv8 e Azure

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Azure-Blob%20Storage-0089D6?style=for-the-badge&logo=microsoftazure" alt="Azure">
  <img src="https://img.shields.io/badge/YOLO-v8-00FFFF?style=for-the-badge" alt="YOLOv8">
</p>

---

## 🎬 Apresentação do TCC

> **Assista à demonstração completa do projeto — do modelo YOLOv8 ao dashboard em produção:**

<p align="center">
  <a href="https://youtu.be/COjS3xzVBGE" target="_blank">
    <img src="https://img.youtube.com/vi/COjS3xzVBGE/maxresdefault.jpg"
         alt="Apresentação TCC — Detecção Automatizada de Buracos em Vias Urbanas"
         width="85%" />
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/COjS3xzVBGE" target="_blank">
    <img src="https://img.shields.io/badge/▶%20Assistir%20Apresentação%20Completa-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir no YouTube">
  </a>
</p>

<p align="center">
  <em>Pipeline de inferência, dashboard interativo e arquitetura Azure em ação</em>
</p>

---

## 🎓 Sobre o Projeto

Este projeto é um **Trabalho de Conclusão de Curso (TCC)** em Sistemas de Informação/Ciência da Computação pela **Universidade Paulista (UNIP)**.

O sistema propõe uma solução inteligente para o monitoramento de vias urbanas, utilizando redes neurais convolucionais para identificar buracos e anomalias no asfalto em tempo real, enviando os dados para uma infraestrutura escalável na nuvem.

### 🧠 Inteligência Artificial

- **Modelo:** YOLOv8 (You Only Look Once)
- **Performance:** mAP@50 de **78%**
- **Classes:** `pothole` (buraco)
- **Pipeline:** Recebe imagem → Inferência → Marcação de Bounding Boxes → Persistência

### ☁️ Arquitetura Cloud (Azure)

O sistema foi desenhado para ser resiliente e escalável:

1. **Backend:** FastAPI rodando em VM Linux (Ubuntu 24.04)
2. **Storage:** Azure Blob Storage para armazenamento das evidências (fotos originais e anotadas)
3. **Database:** SQLite para metadados e logs de detecção
4. **Frontend:** Dashboard interativo em tempo real

---

## 🚀 Como Executar

### 1. Requisitos

- Python 3.12+
- Conta na Azure (opcional, para armazenamento em nuvem)

### 2. Instalação

```bash
# Clonar o repositório
git clone https://github.com/alexandrade-git/PootholeTCC.git
cd PootholeTCC

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 3. Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
API_KEY=sua_chave_secreta
AZURE_CONN_STR=sua_connection_string
AZURE_CONTAINER=detections
```

---

## 📊 Endpoints Principais

| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `POST` | `/predict/base64` | Envia imagem em Base64 para detecção. |
| `GET` | `/dashboard` | Interface visual de monitoramento. |
| `GET` | `/api/insights` | Dados estatísticos das últimas 24h. |
| `GET` | `/health` | Check de status do modelo e sistema. |

---

## 📸 Dashboard

O dashboard foi desenvolvido com foco na usabilidade para gestores públicos, permitindo visualizar a confiança da IA e a localização (estimada) da ocorrência.

---

## 👥 Autores

- **Alex Ryan Andrade de Oliveira**
- Felipe Correia de Oliveira
- Gustavo Silva Vieira
- Wilham de Deus Ferreira
- Leonardo Afonso Dinareli

**Instituição:** UNIP - Universidade Paulista — 2026
