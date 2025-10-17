"""
Envio inteligente de imagens a partir de uma câmera de monitoramento (ex: DroidCam, webcam, RTSP).

Funcionalidades:
- Captura de fonte (RTSP URL ou device index)
- Envio periódico ou por detecção de movimento
- Opção para ignorar movimento e enviar apenas periodicamente
"""

import cv2
import argparse
import time
import base64
import requests
import os
import sys
import urllib.parse
import logging

DEFAULT_URL = 'http://4.227.239.90:8000/predict/file?cam_id=cam1'
DEFAULT_KEY = None


def send_file_bytes(url, api_key, img_bytes, cam_id='camera', filename='frame.jpg'):
    """Envia a imagem como multipart/form-data para /predict/file com o parâmetro cam_id."""
    parsed = urllib.parse.urlparse(url)
    if parsed.path.endswith('/predict/file'):
        endpoint = url
    else:
        endpoint = url.rstrip('/') + '/predict/file'
    headers = {'X-API-KEY': api_key}
    params = {'cam_id': cam_id}
    files = {'file': (filename, img_bytes, 'image/jpeg')}
    return requests.post(endpoint, headers=headers, params=params, files=files, timeout=30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=0, help='RTSP URL ou device index (0 padrão)')
    p.add_argument('--url', default=DEFAULT_URL, help='URL base da API FastAPI')
    p.add_argument('--cam-id', default='cam1', help='ID da câmera (cam_id)')
    p.add_argument('--key', default=DEFAULT_KEY, help='API key (ou defina a variável de ambiente API_KEY)')
    p.add_argument('--periodic-interval', type=int, default=21600, help='Enviar um frame a cada N segundos (padrão: 6h)')
    p.add_argument('--ignore-motion', action='store_true', default=True, help='Ignorar detecção de movimento e enviar apenas periodicamente')
    p.add_argument('--save-dir', default=None, help='Salvar imagens localmente')
    p.add_argument('--log-file', default='send_from_camera.log', help='Arquivo de log')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, format='%(asctime)s %(levelname)s: %(message)s')

    # Preparar fonte
    try:
        source = int(args.source)
    except Exception:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error('Não foi possível abrir a fonte: %s', args.source)
        sys.exit(1)

    # Esperar DroidCam iniciar
    if source == 1:
        logging.info('Detectado source=1 (DroidCam). Aguardando inicialização da câmera...')
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if ret and frame is not None and frame.sum() > 0:
                logging.info('Câmera inicializada após %.1f segundos.', time.time() - start_time)
                # descartar frames iniciais (pretos)
                for _ in range(10):
                    cap.read()
                break
            if time.time() - start_time > 30:
                logging.error('Timeout: DroidCam (source=1) não iniciou após 30 segundos.')
                sys.exit(1)
            time.sleep(0.5)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    last_sent = 0
    logging.info('Iniciando captura. Envio a cada %.1f horas.', args.periodic_interval / 3600)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(1)
                continue

            now = time.time()
            if now - last_sent >= args.periodic_interval:
                # codifica e envia
                _, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                img_bytes = jpg.tobytes()
                filename = f'detected_{int(now)}.jpg'

                if args.save_dir:
                    path = os.path.join(args.save_dir, filename)
                    with open(path, 'wb') as f:
                        f.write(img_bytes)
                    logging.info('Imagem salva localmente: %s', path)

                logging.info('Enviando frame periódico para API (%s)', args.url)
                try:
                    r = send_file_bytes(args.url, args.key, img_bytes, cam_id=args.cam_id, filename=filename)
                    logging.info('Resposta da API: %s', r.status_code)
                except Exception as e:
                    logging.exception('Erro ao enviar frame')

                last_sent = now

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
    finally:
        cap.release()


if __name__ == '__main__':
    main()
