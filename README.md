# Protease Cleavage Prediction Webtool

Web tool for background protease substrate prediction workflows, with Google login, job tracking, and downloadable results.

## Main folders
- `frontend/` — web app (React + Vite)
- `backend/` — API (FastAPI)
- `worker/` — background analysis pipeline (FastAPI)
- `shared/` — shared Python modules
- `docs/` — planning and architecture

---

## Running locally

### Option A — Docker Compose (Windows / Mac / Linux)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows users: enable WSL2 backend during installation)

```bash
# Clone the repo and start everything
git clone https://github.com/pedronobrez/CPRED-local.git
cd CPRED-local
docker compose up --build
```

Services:
| Service  | URL                    |
|----------|------------------------|
| Frontend | http://localhost:5173  |
| Backend  | http://localhost:8080  |
| Worker   | http://localhost:8001  |

Authentication is bypassed in local mode — no Firebase account needed.

To stop: `Ctrl+C`, then `docker compose down`.

---

### Option B — Native (Linux / macOS only)

**Prerequisites:** Python 3.12+, [uv](https://astral.sh/uv), Node.js 20+

```bash
./start-local.sh
```
