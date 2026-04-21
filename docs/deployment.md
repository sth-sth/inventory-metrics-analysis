# Deployment Guide (Production Web)

## Option A: Streamlit Community Cloud (Fastest)

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select this repo and set main file to `app.py`.
4. Deploy.

Notes:
- `requirements.txt` and `.streamlit/config.toml` are already configured.
- Default view uses built-in demo data, so visitors can see the product instantly.

## Option B: Render (Docker, Recommended for Team Demo)

1. Push repository to GitHub.
2. In Render, create a new Web Service from this repo.
3. Render detects `render.yaml` and `Dockerfile` automatically.
4. Deploy and wait for build.

## Option C: Any Docker Platform (Railway/Fly.io/VM/Kubernetes)

Build and run locally:

```bash
docker build -t inventory-intelligence .
docker run -p 8501:8501 inventory-intelligence
```

Then open `http://localhost:8501`.

## Recommended Production Hardening

- Add authentication/SSO for private business data.
- Enable HTTPS termination on platform ingress.
- Add object storage or database for upload history.
- Add scheduled refresh and alert push integrations.
- Add structured logging and uptime monitoring.
