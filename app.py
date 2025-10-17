from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import io
from PIL import Image
import time
import os
import numpy as np
import torch
from ultralytics import YOLO
import json
import datetime
from pathlib import Path
import logging

# DB / Azure
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from azure.storage.blob import BlobServiceClient

MODEL_PATH = "best.pt"  # preferível caminho relativo dentro do projeto
DETECTIONS_DIR = "detections"
SAVE_DIR = "received_frames"
os.makedirs(DETECTIONS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Seleciona device automaticamente
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# limiar de confiança para inferência (ajustado)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))

# lazy-load do modelo
_model = None
def load_model_if_exists():
    global _model
    model_path = os.getenv("MODEL_PATH", MODEL_PATH)
    if _model is None:
        if not os.path.exists(model_path):
            return None
        _model = YOLO(model_path)
    return _model

app = FastAPI(title="Camera Receiver API")
templates = Jinja2Templates(directory="templates")

# servir imagens locais (recebidas e anotadas)
app.mount("/detections", StaticFiles(directory=os.path.abspath(DETECTIONS_DIR)), name="detections")
app.mount("/received_frames", StaticFiles(directory=os.path.abspath(SAVE_DIR)), name="received_frames")


# read API_KEY from env (fallback to None)
API_KEY = os.getenv("API_KEY")

# Azure config via env vars (safer). Se não definido, permanece None e upload para blob será ignorado.
AZURE_CONN_STR = os.getenv("AZURE_CONN_STR")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "detections")

# Database config
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "app.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    cam_id = Column(String, index=True)
    timestamp = Column(DateTime)
    saved_local = Column(String, nullable=True)
    saved_blob = Column(String, nullable=True)
    annotated_local = Column(String, nullable=True)
    annotated_blob = Column(String, nullable=True)
    detections_json = Column(Text)
    max_confidence = Column(Float)
    mean_confidence = Column(Float)
    buraco_detected = Column(Boolean)
    # coluna adicional (pt-br) existente no banco; manter para compatibilidade
    buraco_detectado = Column(Integer, nullable=True)


@app.on_event("startup")
def on_startup():
    """Inicializações a executar quando o app iniciar: testar conexão e criar tabelas."""
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Usando DATABASE_URL={DATABASE_URL}")
    try:
        with engine.connect() as conn:
            logging.info("Conexão com o banco de dados bem-sucedida.")
        Base.metadata.create_all(bind=engine)
        logging.info("Tabelas criadas com sucesso no banco de dados SQLite.")
    except Exception as e:
        logging.exception(f"Erro ao configurar o banco de dados: {e}")

# Azure client lazy
_blob_service = None
def get_blob_service():
    global _blob_service
    if _blob_service is None and AZURE_CONN_STR:
        try:
            _blob_service = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
            try:
                _blob_service.create_container(AZURE_CONTAINER)
                logging.info('Container ensured: %s', AZURE_CONTAINER)
            except Exception as e:
                logging.warning('Não foi possível criar/assegurar container %s: %s', AZURE_CONTAINER, e)
        except Exception as e:
            logging.exception('Erro ao instanciar BlobServiceClient: %s', e)
            _blob_service = None
    return _blob_service

class ImagePayload(BaseModel):
    b64: str

def _save_bytes_to_file(b: bytes, cam_id: str, prefix: str = "frame"):
    ts = int(time.time())
    name = f"{prefix}_{cam_id}_{ts}.jpg"
    path = os.path.join(SAVE_DIR, name)
    try:
        with open(path, "wb") as f:
            f.write(b)
    except Exception:
        logging.exception('Falha salvando localmente %s', path)

    blob_url = None
    if AZURE_CONN_STR:
        svc = get_blob_service()
        if svc:
            try:
                logging.info('Tentando upload para blob: %s (container=%s)', name, AZURE_CONTAINER)
                blob_client = svc.get_blob_client(container=AZURE_CONTAINER, blob=name)
                blob_client.upload_blob(b, overwrite=True)
                blob_url = blob_client.url
                logging.info('Upload bem-sucedido: %s', blob_url)
            except Exception as e:
                logging.exception('Erro ao fazer upload do blob %s: %s', name, e)
                blob_url = None

    return path, blob_url

def _run_inference_on_numpy(img_bgr):
    """
    Recebe imagem numpy BGR, retorna (detections_list, annotated_numpy_or_None).
    detections_list -> list of {box, confidence, class, label}
    """
    model = load_model_if_exists()
    if model is None:
        return None, None  # sinaliza que modelo não está disponível

    # passar limiar de confiança explícito
    results = model.predict(img_bgr, imgsz=640, device=DEVICE, verbose=False, conf=CONFIDENCE_THRESHOLD)
    res0 = results[0]
    boxes = getattr(res0.boxes, "xyxy", None)
    confs = getattr(res0.boxes, "conf", None)
    clss = getattr(res0.boxes, "cls", None)
    names = getattr(res0, "names", {}) or {}

    detections = []
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            try:
                box = [float(x) for x in boxes[i]]
                conf = float(confs[i]) if confs is not None else None
                cls = int(clss[i]) if clss is not None else None
            except Exception:
                box = [float(x) for x in boxes[i].tolist()]
                conf = float(confs[i].tolist()) if confs is not None else None
                cls = int(clss[i].tolist()) if clss is not None else None
            label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
            detections.append({
                "box": box,
                "confidence": conf,
                "class": cls,
                "label": label
            })

    # annotated image (numpy BGR) - pode falhar em alguns modelos; tratamos com try/except
    annotated = None
    try:
        annotated = res0.plot()
    except Exception:
        annotated = None

    return detections, annotated

@app.get("/health")
async def health():
    model = load_model_if_exists()
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH if model is not None else None}

@app.post("/predict/base64")
async def predict_base64(request: Request, payload: ImagePayload):
    # valida API key
    key = request.headers.get("X-API-KEY")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    cam_id = request.query_params.get("cam_id", "camera")
    timestamp = int(time.time())

    # decodifica
    try:
        img_bytes = base64.b64decode(payload.b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 inválido: {e}")

    # salva original (local + opcional blob)
    saved_path = None
    saved_blob = None
    try:
        saved_path, saved_blob = _save_bytes_to_file(img_bytes, cam_id, prefix="received")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar imagem: {e}")

    # prepara numpy BGR para inferência
    try:
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)[:, :, ::-1]  # RGB -> BGR
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar imagem: {e}")

    detections, annotated = _run_inference_on_numpy(img_np)

    model_loaded = detections is not None
    buraco_detectado = False
    annotated_path = None
    annotated_blob = None

    if not model_loaded:
        message = "Modelo não encontrado no servidor; imagem salva, sem inferência."
    else:
        buraco_detectado = len(detections) > 0
        if annotated is not None and buraco_detectado:
            try:
                os.makedirs(DETECTIONS_DIR, exist_ok=True)
                annotated_name = f"annotated_{cam_id}_{timestamp}.jpg"
                annotated_path = os.path.join(DETECTIONS_DIR, annotated_name)
                import cv2
                cv2.imwrite(annotated_path, annotated)
                # enviar também para blob
                try:
                    with open(annotated_path, "rb") as af:
                        _, annotated_blob = _save_bytes_to_file(af.read(), cam_id, prefix=f"annotated_{timestamp}")
                except Exception:
                    annotated_blob = None
            except Exception:
                annotated_path = None
        message = "Inferência executada" if model_loaded else "Modelo ausente"
    # calcular métricas de confiança (máxima e média) a partir das detecções
    confidences = [float(d.get('confidence')) for d in (detections or []) if d.get('confidence') is not None]
    if confidences:
        max_conf = max(confidences)
        mean_conf = sum(confidences) / len(confidences)
    else:
        max_conf = 0.0
        mean_conf = 0.0

    # persistir registro no DB
    try:
        sess = SessionLocal()
        # gravar ambos os campos (bool e inteiro) para compatibilidade com o esquema
        rec = Detection(
            cam_id=cam_id,
            timestamp=datetime.datetime.fromtimestamp(timestamp),
            saved_local=saved_path,
            saved_blob=saved_blob,
            annotated_local=annotated_path,
            annotated_blob=annotated_blob,
            detections_json=json.dumps(detections or []),
            max_confidence=max_conf,
            mean_confidence=mean_conf,
            buraco_detected=bool(buraco_detectado),
            buraco_detectado=1 if buraco_detectado else 0,
        )
        sess.add(rec)
        sess.commit()
        sess.close()
    except Exception as e:
        logging.exception('Erro ao persistir registro no DB: %s', e)

    response = {
        "detections": detections or [],
        "saved_file": saved_path,
        "saved_file_blob": saved_blob,
        "annotated_file": annotated_path,
        "annotated_file_blob": annotated_blob,
        "cam_id": cam_id,
        "timestamp": timestamp,
        "buraco_detectado": bool(buraco_detectado),
        "model_loaded": bool(model_loaded),
        "max_confidence": round(max_conf, 3),
        "mean_confidence": round(mean_conf, 3),
        "message": message
    }
    return JSONResponse(response)

@app.post("/predict/file")
async def predict_file(request: Request, file: UploadFile = File(...)):
    # valida API key
    key = request.headers.get("X-API-KEY")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    cam_id = request.query_params.get("cam_id", "camera")
    timestamp = int(time.time())

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro lendo arquivo: {e}")

    # salva original
    try:
        saved_path, saved_blob = _save_bytes_to_file(content, cam_id, prefix="received")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar imagem: {e}")

    # prepara numpy BGR
    try:
        img_pil = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(img_pil)[:, :, ::-1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar imagem: {e}")

    detections, annotated = _run_inference_on_numpy(img_np)

    model_loaded = detections is not None
    buraco_detectado = False
    annotated_path = None
    annotated_blob = None

    if not model_loaded:
        message = "Modelo não encontrado no servidor; imagem salva, sem inferência."
    else:
        buraco_detectado = len(detections) > 0
        if annotated is not None and buraco_detectado:
            try:
                os.makedirs(DETECTIONS_DIR, exist_ok=True)
                annotated_name = f"annotated_{cam_id}_{timestamp}.jpg"
                annotated_path = os.path.join(DETECTIONS_DIR, annotated_name)
                import cv2
                cv2.imwrite(annotated_path, annotated)
                try:
                    with open(annotated_path, "rb") as af:
                        _, annotated_blob = _save_bytes_to_file(af.read(), cam_id, prefix=f"annotated_{timestamp}")
                except Exception:
                    annotated_blob = None
            except Exception:
                annotated_path = None
        message = "Inferência executada"

    # persistir registro
    try:
        sess = SessionLocal()
        max_conf_val = max([float(d.get('confidence')) for d in (detections or [])]) if detections else 0.0
        mean_conf_val = (sum([float(d.get('confidence')) for d in (detections or [])]) / len(detections)) if detections else 0.0
        rec = Detection(
            cam_id=cam_id,
            timestamp=datetime.datetime.fromtimestamp(timestamp),
            saved_local=saved_path,
            saved_blob=saved_blob,
            annotated_local=annotated_path,
            annotated_blob=annotated_blob,
            detections_json=json.dumps(detections or []),
            max_confidence=max_conf_val,
            mean_confidence=mean_conf_val,
            buraco_detected=bool(buraco_detectado),
            buraco_detectado=1 if buraco_detectado else 0,
        )
        sess.add(rec)
        sess.commit()
        sess.close()
    except Exception as e:
        logging.exception('Erro ao persistir registro no DB: %s', e)

    response = {
        "detections": detections or [],
        "saved_file": saved_path,
        "saved_file_blob": saved_blob,
        "annotated_file": annotated_path,
        "annotated_file_blob": annotated_blob,
        "cam_id": cam_id,
        "timestamp": timestamp,
        "buraco_detectado": bool(buraco_detectado),
        "model_loaded": bool(model_loaded),
        "message": message
    }
    return JSONResponse(response)


# ...existing code...
@app.get("/api/insights")
def api_insights(request: Request, period: str = "24h", show_all: bool = False):
    try:
        hours = int(period.replace("h", ""))
    except Exception:
        hours = 24
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)

    sess = SessionLocal()
    try:
        rows = sess.query(Detection).filter(Detection.timestamp >= since).order_by(Detection.timestamp.desc()).all()

        detections = []
        images = []
        by_device = {}

        for r in rows:
            cam_id = r.cam_id
            ts = r.timestamp

            # parse detections_json and derive detection flags/metrics
            try:
                det_list = json.loads(r.detections_json) if r.detections_json else []
            except Exception:
                det_list = []
            buraco = True if det_list else False
            max_conf = max((d.get("confidence", 0.0) for d in det_list), default=0.0)
            mean_conf = (sum((d.get("confidence", 0.0) for d in det_list)) / len(det_list)) if det_list else 0.0

            # choose image URL: prefer annotated_blob (blob), then annotated_local, then saved_blob, then saved_local
            img_url = None
            if r.annotated_blob:
                img_url = str(r.annotated_blob)
            elif r.annotated_local:
                fname = os.path.basename(r.annotated_local)
                try:
                    img_url = str(request.url_for("detections", path=fname))
                except Exception:
                    img_url = f"/detections/{fname}"
            elif r.saved_blob:
                img_url = str(r.saved_blob)
            elif r.saved_local:
                fname = os.path.basename(r.saved_local)
                try:
                    img_url = str(request.url_for("received_frames", path=fname))
                except Exception:
                    img_url = f"/received_frames/{fname}"

            # adicionar imagem somente quando houver detecção (ou quando explicitamente solicitado)
            if img_url and (buraco or show_all):
                images.append(img_url)

            # only count detections that actually have boxes (buraco == True)
            if cam_id not in by_device:
                by_device[cam_id] = {"cam_id": cam_id, "count": 0, "max_confidence": 0.0, "avg_confidence_sum": 0.0}
            if buraco:
                by_device[cam_id]["count"] += 1
                by_device[cam_id]["avg_confidence_sum"] += mean_conf
                if max_conf > by_device[cam_id]["max_confidence"]:
                    by_device[cam_id]["max_confidence"] = max_conf

            # incluir este registro na lista de detections apenas se houver boxes (buraco) ou quando show_all=true
            if buraco or show_all:
                detections.append({
                    "id": r.id,
                    "cam_id": cam_id,
                    "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    "max_confidence": float(max_conf),
                    "mean_confidence": float(mean_conf),
                    "image_url": img_url,
                    "saved_blob": r.saved_blob,
                    "saved_local": r.saved_local,
                    "buraco_detectado": buraco
                })

        by_device_list = []
        for cam_id, info in by_device.items():
            count = info["count"]
            avg_conf = (info["avg_confidence_sum"] / count) if count else 0.0
            by_device_list.append({
                "cam_id": cam_id,
                "count": count,
                "avg_confidence": avg_conf,
                "max_confidence": info["max_confidence"]
            })

        by_device_list.sort(key=lambda x: x["count"], reverse=True)

        return {
            "detections": detections,
            "images": images,
            "by_device": by_device_list,
            "total": len(detections),
            "period": period,
            "show_all": bool(show_all)
        }
    except Exception:
        logging.exception("Erro em /api/insights")
        raise HTTPException(status_code=500, detail="Erro interno")
    finally:
        sess.close()
# ...existing code...


@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})