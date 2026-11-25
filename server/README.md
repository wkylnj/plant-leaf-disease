# Backend API

FastAPI + SQLite inference service. The default backend is `auto`: load ONNX, then PyTorch, then the deterministic fake predictor. The fake predictor is for client integration only; its results are not meaningful.

## Run locally and open on iPhone

Run these commands in PowerShell from the repository root. Create the virtual environment and install dependencies only on the first run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r server\requirements.txt
```

Start the service (repeat this part each time you want to use the app):

```powershell
$env:PREDICT_BACKEND = "fake"
$env:SAVE_UPLOADS = "0"
Set-Location server
..\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The root URL serves the mobile web app. On the computer, open `http://127.0.0.1:8000/`. To use it from iPhone Safari:

1. Connect the iPhone and computer to the same Wi-Fi network.
2. On Windows, run `ipconfig` and find the IPv4 address under the active Wi-Fi adapter (for example, `192.168.1.23`).
3. On the iPhone, open `http://<computer IPv4 address>:8000/`, for example `http://192.168.1.23:8000/`.
4. Keep the PowerShell window and service running while using the page. If Windows Firewall prompts you, allow Python/Uvicorn on private networks. If the page opens but cannot reach the API, confirm the address and port, then check that the firewall allows inbound TCP port 8000 on the private network.

The web page derives the API address from the URL used to open it. You can change the backend address in the page if needed. `/docs` provides the interactive API page, and `/health` reports service status. The service creates `server/app.db` on first start. Uploaded photos are not saved by default; set `SAVE_UPLOADS=1` only for local debugging.

`PREDICT_BACKEND=fake` starts without model weights so the mobile workflow can be checked, but its disease predictions are synthetic and must not be used for diagnosis. To run a real model, place `model.onnx` and its matching `classes.json` in `server/models/` and use `PREDICT_BACKEND=onnx` instead. Keep the class order identical to the model output order.

## API

`X-Device-ID` is a random, app-generated identifier. It is not a login credential and contains no personal information. It scopes a device's history and feedback. `POST /predict` accepts it optionally for compatibility; history APIs require it.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service status, selected backend, version and class count |
| `GET` | `/classes` | Model class IDs and Chinese crop/disease names |
| `POST` | `/predict` | Multipart image in field `file`; returns request ID and Top-3 |
| `POST` | `/feedback` | Save correctness and optional corrected class for a prediction |
| `GET` | `/history?limit=50&offset=0` | List this device's predictions with pagination and total count |
| `GET` | `/history/{request_id}` | Read one of this device's records, including feedback state |
| `DELETE` | `/history` | Delete this device's records and feedback |
| `GET` | `/stats` | Aggregate prediction and feedback counts |
| `GET` | `/recent?limit=20` | Recent aggregate records for local administration |
| `POST` | `/users/{device_id}` | Register an anonymous device for user counts |

Example:

```powershell
$device = "demo-device-id-123456"
curl.exe -H "X-Device-ID: $device" -F "file=@D:\photos\leaf.jpg" http://127.0.0.1:8000/predict
curl.exe -H "X-Device-ID: $device" http://127.0.0.1:8000/history
```

`/history` returns `{ "items": [...], "total": 12, "limit": 50, "offset": 0 }`. `/predict` enforces the configured 8 MiB upload limit and accepts JPEG/PNG. Its `request_id` is the ID used by `/feedback` and `/history/{request_id}`. A feedback request from one device cannot modify another device's prediction when the device header is supplied.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PREDICT_BACKEND` | `auto` | `auto`, `onnx`, `torch` or `fake` |
| `DB_PATH` | `server/app.db` | SQLite database file |
| `UPLOAD_DIR` | `server/uploads` | Optional upload directory |
| `SAVE_UPLOADS` | `0` | Set to `1` to retain uploaded images on the server |
| `MAX_UPLOAD_BYTES` | `8388608` | Maximum request image size |
| `PORT` | `8000` | Port when starting with `python main.py` |

Existing SQLite databases are migrated on startup by adding the nullable `device_id` column. Older records without a device ID remain available through `/recent`, but are not exposed in per-device history.
