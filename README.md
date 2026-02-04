# Yolo + Ray Serve

This project exposes a YOLO inference pipeline through Ray Serve. The HTTP ingress receives a multipart form with a single file field named `frames`, decodes it, runs detection, and returns JSON.

## Repository layout

- [server/src](server/src) contains Ray Serve deployments (`ingress.py`, `image_decoder.py`, `yolo_detector.py`) and the app entrypoint (`main_service.py`).
- [server/requirements.txt](server/requirements.txt) contains server dependencies.
- [client/src](client/src) contains the test client.
- [k8s](k8s) contains Kubernetes manifests and image references.

## Server setup (local)

Install dependencies from the server folder, then start Ray Serve:

- Install: `pip install -r server/requirements.txt`
- Run: `serve run main_service:app`

The Ray Serve HTTP endpoint is the default Serve address (e.g. `http://127.0.0.1:8000`).

## HTTP endpoint

Send a `multipart/form-data` request with a single file field named `frames`:

- Endpoint: `/`
- Method: `POST`
- Body: `form-data` with key `frames` (type **File**)

The response is JSON containing detections and timing information.

## Client

From [client/src](client/src):

- Run: `python3 client.py`

The main function is:

`process_image_repeatedly(20, 600, 'http://ip:port/')`

Arguments:

- fps (target framerate)
- duration (seconds)
- server URL (Ray Serve ingress)

The client writes a `result.dat` file with expected and measured framerate.

## Kubernetes

Manifests are under [k8s](k8s). Two images are available:

- `stegala/yolo-server:tiny`
- `stegala/yolo-server:reg`

Update the image in the manifests before deploying.

