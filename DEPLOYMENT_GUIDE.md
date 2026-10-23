# Pomegranate-MultiMed AI: Production Deployment Guide

This guide provides end-to-end instructions for deploying the **Pomegranate-MultiMed AI** diagnostic system across different production environments:
1. **Free Cloud Hosting (Hugging Face Spaces / Render)**
2. **Docker Container Deployment (AWS, GCP, DigitalOcean, Any VPS)**
3. **Instant Public Web Tunnel (ngrok / Cloudflare Tunnel)**
4. **Local Network / Wi-Fi Access (Farm & Clinic Field Use)**

---

## Architecture & Deployment Artifacts

The deployment bundle includes:
- [`Dockerfile`](Dockerfile) — Lightweight multi-stage container image with CPU-optimized PyTorch.
- [`docker-compose.yml`](docker-compose.yml) — Single-command container orchestrator.
- [`requirements.txt`](requirements.txt) — Locked production dependencies.
- [`start_production.bat`](start_production.bat) — One-click Windows local production launcher.
- [`app/server.py`](app/server.py) — Production HTTP & REST API server (binds to `0.0.0.0` and supports dynamic `$PORT`).

---

## Option 1: Free Cloud Deployment (Hugging Face Spaces - Docker)

Hugging Face Spaces provides **free persistent hosting** with 16GB RAM and 2 vCPUs:

1. **Create a Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Space Name: `pomegranate-multimed-ai`
   - License: `MIT`
   - Space SDK: Select **Docker** (Blank template).
   - Space Visibility: **Public**.

2. **Clone and Push Code**:
   ```bash
   git clone https://huggingface.co/spaces/<YOUR-USERNAME>/pomegranate-multimed-ai
   cd pomegranate-multimed-ai
   ```
   Copy the following files into the cloned repository:
   - `Dockerfile`
   - `requirements.txt`
   - `app/`
   - `outputs/baseline/`
   - `xai/`

3. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Deploy Pomegranate-MultiMed AI"
   git push
   ```
   Hugging Face will automatically build the Docker container and provide a permanent public HTTPS URL (e.g., `https://<YOUR-USERNAME>-pomegranate-multimed-ai.hf.space`).

---

## Option 2: Cloud Deployment on Render / Railway

### Deploying to Render (Free Tier):
1. Push this repository to GitHub or GitLab.
2. Sign in to [render.com](https://render.com) and click **New +** $\to$ **Web Service**.
3. Connect your repository.
4. Set Environment to **Docker**.
5. Render automatically detects the `Dockerfile` and binds to `$PORT`.
6. Click **Deploy Web Service**. You will receive an `onrender.com` HTTPS URL.

---

## Option 3: Docker Container Deployment (AWS, GCP, or VPS)

To run anywhere with Docker installed:

### Build and Run with Docker Compose:
```bash
docker-compose up -d --build
```

### Or using standard Docker CLI:
```bash
# 1. Build the Docker image
docker build -t pomegranate-multimed-ai .

# 2. Run the container
docker run -d -p 8080:8080 --name pomegranate-ai --restart unless-stopped pomegranate-multimed-ai
```

The system is now live at `http://<SERVER_IP>:8080`.

---

## Option 4: Instant Public Tunnel (Share Live Right Now)

If your local server is already running (`python app/server.py 8080`), you can share a live public link instantly using a tunnel:

### Using Cloudflare Tunnel (Free, no account needed):
```bash
npx untun@latest tunnel http://localhost:8080
```
This generates an instant secure `https://*.trycloudflare.com` link that anyone in the world can access.

### Using ngrok:
```bash
ngrok http 8080
```

---

## Option 5: Local Area Network (LAN) / Field Mobile Access

Because `app/server.py` binds to `0.0.0.0`, any smartphone, iPad, or computer connected to the same Wi-Fi / Hotspot can use the application directly:

1. **Find your local IP address**:
   - In PowerShell: `ipconfig`
   - Look for **IPv4 Address** under your active Wi-Fi or Ethernet adapter (e.g., `192.168.1.45`).

2. **Access from phone or tablet**:
   - Open browser on your phone and navigate to:
     `http://192.168.1.45:8080`
   - You can take live photos of pomegranate leaves and fruits using your phone's camera, run the diagnosis, and download the PDF report on mobile.
