# 🐳 Docker Guide — Drowsiness Detection

Complete guide to containerising and running the Drowsiness Detection system using Docker — from prerequisites to daily operations.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Docker Files](#2-project-docker-files)
3. [Fix Docker Permissions](#3-fix-docker-permissions)
4. [Allow X11 Display](#4-allow-x11-display)
5. [Build the Image](#5-build-the-image)
6. [Run the Container](#6-run-the-container)
7. [Environment Variables](#7-environment-variables)
8. [Volume Mounts](#8-volume-mounts)
9. [Docker Compose (Recommended)](#9-docker-compose-recommended)
10. [Multi-Architecture Build](#10-multi-architecture-build)
11. [Useful Docker Commands](#11-useful-docker-commands)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

Ensure the following are installed on your host machine before proceeding.

### Docker Engine

```bash
# Check if Docker is installed
docker --version
# Expected: Docker version 29.x.x or higher

# Check Docker daemon is running
docker info | head -5
```

If not installed, follow the official guide: https://docs.docker.com/engine/install/ubuntu/

### Docker Compose (Plugin)

```bash
docker compose version
# Expected: Docker Compose version v2.x.x
```

### System Requirements

| Requirement | Minimum | Notes |
|------------|---------|-------|
| OS | Ubuntu 20.04+ | Or any Linux with X11 |
| RAM | 4 GB | 6–8 GB recommended |
| CPU | 4 cores | NCNN uses OpenMP multi-threading |
| Disk | 10 GB free | Image is ~3–4 GB after build |
| Python (host) | 3.12 | Only needed for local dev |
| Camera | USB `/dev/video0` or RTSP URL | — |

---

## 2. Project Docker Files

Three files were added to the project root:

```
drowsiness_detection/
├── Dockerfile           ← Multi-stage image build
├── docker-compose.yml   ← Easy one-command run with all flags pre-configured
└── .dockerignore        ← Keeps build context lean (excludes logs, cache, etc.)
```

### `Dockerfile` — What it does

The Dockerfile uses a **two-stage build** to keep the final image as small as possible:

```
Stage 1 — builder (python:3.12-slim)
  ├── Installs build tools: gcc, g++, cmake
  ├── Installs system libs needed at compile time
  └── Runs: pip install -r requirements.txt → /install

Stage 2 — runtime (python:3.12-slim)
  ├── Installs ONLY runtime system libs (no compilers)
  ├── Copies /install from builder (pre-built Python wheels)
  ├── Copies: main.py, src/, models/
  ├── Creates non-root user: drowsy
  └── ENTRYPOINT: python main.py
```

#### Runtime system libraries installed

| Library | Purpose |
|---------|---------|
| `libgl1` | OpenCV — libGL at runtime |
| `libglib2.0-0` | MediaPipe / GLib |
| `libsm6`, `libxext6` | X11 extensions for OpenCV window |
| `libxrender1` | X rendering for OpenCV display |
| `libgomp1` | OpenMP — NCNN multi-threaded inference |
| `libsdl2-2.0-0` | SDL2 audio backend for pygame alerts |
| `pulseaudio-utils` | PulseAudio client tools for host audio |
| `libgstreamer1.0-0` | OpenCV camera back-end support |
| `v4l-utils` | USB webcam debugging tools |

### `.dockerignore` — What is excluded from build context

```
.git/           # Git history
.vscode/        # IDE settings
__pycache__/    # Python cache
logs/           # Mounted as volume at runtime
employee/       # Mounted as volume at runtime
test/           # Test files
*.pyc, *.csv    # Compiled Python / CSV log files
test.jpg        # Test images
compare.py      # Development script
```

---

## 3. Fix Docker Permissions

By default, only `root` can talk to the Docker socket. Run this **once** to add your user to the `docker` group:

```bash
sudo usermod -aG docker $USER
```

Apply the change **without logging out** (for the current terminal only):

```bash
newgrp docker
```

> **Note:** For a permanent fix across all terminals, fully log out and log back in.

Verify it works:

```bash
docker info | head -5
# Should show client version without any "permission denied" error
```

---

## 4. Allow X11 Display

The container needs permission to draw OpenCV windows on your host desktop. Run this **once per session** (after each login):

```bash
xhost +local:docker
```

Expected output:
```
non-network local connections being added to access control list
```

> This allows any local Docker container to connect to your X server. Revoke it after use with `xhost -local:docker` if needed.

---

## 5. Build the Image

```bash
# Make sure you are in the project root
cd ~/Music/drowsiness_detection

# Build the image (takes 5–15 min on first run — downloads ~700 MB of packages)
docker build -t drowsiness-detection .
```

### What happens during build

```
Step 1/18  FROM python:3.12-slim          ← pulls base image
Step 2/18  RUN apt-get install ...        ← installs build tools (builder stage)
Step 3/18  WORKDIR /build
Step 4/18  COPY requirements.txt
Step 5/18  RUN pip install ...            ← installs all Python packages (~5–10 min)
Step 6/18  FROM python:3.12-slim          ← fresh runtime base
Step 7/18  RUN apt-get install ...        ← installs runtime libs only
Step 8/18  COPY --from=builder /install   ← copies compiled wheels
Step 9–13  COPY source files
Step 14    RUN mkdir logs employee
Step 15    RUN useradd drowsy             ← creates non-root user
Step 16    USER drowsy
Step 17    ENV RTSP_URL DISPLAY ...
Step 18    ENTRYPOINT python main.py
Successfully built <image_id>
Successfully tagged drowsiness-detection:latest  ✅
```

### Verify image was built

```bash
docker images drowsiness-detection
# REPOSITORY              TAG       IMAGE ID       CREATED         SIZE
# drowsiness-detection    latest    ba0ccf23b40c   2 minutes ago   ~3.5GB
```

---

## 6. Run the Container

### Option A — USB Webcam (default)

```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/employee:/app/employee \
  --network host \
  drowsiness-detection
```

### Option B — RTSP IP Camera

```bash
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -e RTSP_URL="rtsp://user:pass@192.168.0.119:554/stream" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/employee:/app/employee \
  --network host \
  drowsiness-detection
```

> When `RTSP_URL` is set, the app uses the IP camera. When it is empty, it falls back to `/dev/video0` (USB webcam).

### Option C — Headless (no display, no audio)

```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/employee:/app/employee \
  drowsiness-detection
```

> Audio and display will degrade gracefully — the detector still runs and writes logs.

---

## 7. Environment Variables

All variables can be passed with `-e VAR=value` or set in `docker-compose.yml`.

| Variable | Default | Description |
|----------|---------|-------------|
| `RTSP_URL` | `""` (empty) | RTSP stream URL. Empty = use USB webcam |
| `DISPLAY` | `:0` | X11 display for OpenCV window |
| `PULSE_SERVER` | `unix:/run/user/1000/pulse/native` | PulseAudio socket path |
| `PYTHONUNBUFFERED` | `1` | Streams Python logs to terminal in real-time |

### Switching camera mode via env var

```bash
# USB webcam (default — RTSP_URL is empty)
docker run ... drowsiness-detection

# RTSP camera
docker run ... -e RTSP_URL="rtsp://admin:pass@192.168.1.10:554/cam" drowsiness-detection
```

> `RTSP_URL` is read by `src/config.py` via `os.environ.get("RTSP_URL", "")` so the switch is seamless.

---

## 8. Volume Mounts

| Host Path | Container Path | Mode | Purpose |
|-----------|---------------|------|---------|
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | `rw` | X11 display socket (OpenCV window) |
| `/run/user/1000/pulse` | `/run/user/1000/pulse` | `ro` | PulseAudio socket (alert sounds) |
| `./logs` | `/app/logs` | `rw` | Alert JSONL + frame CSV logs (persisted on host) |
| `./employee` | `/app/employee` | `rw` | Employee face-recognition database |

> **Important:** `logs/` and `employee/` are **not** baked into the image. They are always mounted from the host so data persists across container restarts.

---

## 9. Docker Compose (Recommended)

Using `docker compose` is the easiest way to run with all flags pre-configured.

### Start

```bash
docker compose up
```

### Start in background (detached)

```bash
docker compose up -d
```

### View live logs

```bash
docker compose logs -f
```

### Stop

```bash
docker compose down
```

### Rebuild image and start

```bash
docker compose up --build
```

### Start with RTSP camera

```bash
RTSP_URL="rtsp://admin:pass@192.168.0.119:554/stream" docker compose up
```

### `docker-compose.yml` key settings explained

```yaml
devices:
  - /dev/video0:/dev/video0     # USB webcam passthrough

volumes:
  - /tmp/.X11-unix:/tmp/.X11-unix:rw   # X11 display
  - /run/user/1000/pulse:/run/user/1000/pulse:ro  # PulseAudio
  - ./logs:/app/logs            # persistent logs on host
  - ./employee:/app/employee    # persistent face DB on host

network_mode: host              # allows RTSP LAN access without port mapping

restart: unless-stopped         # auto-restart on crash or reboot
```

---

## 10. Multi-Architecture Build

To build an image that runs on both **x86_64** (PC/server) and **ARM64** (Raspberry Pi, Jetson):

```bash
# Step 1: Install buildx (if not already available)
docker buildx version

# Step 2: Create a multi-arch builder (one-time setup)
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# Step 3: Build and push for both platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/vishnualgosium/drowsiness_detection:latest \
  --push .
```

> Requires a Docker Hub or GitHub Container Registry (GHCR) account to push.
> ARM64 build will be slower as it cross-compiles Python packages.

---

## 11. Useful Docker Commands

### Image management

```bash
# List all images
docker images

# Check image size and layers
docker history drowsiness-detection

# Remove the image
docker rmi drowsiness-detection

# Remove all unused images (free disk space)
docker image prune -a
```

### Container management

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Open a shell inside the running container (for debugging)
docker exec -it <container_id> bash

# Stop a running container
docker stop <container_id>

# Remove all stopped containers
docker container prune
```

### Inspect and debug

```bash
# Check what webcam devices are available on host
ls /dev/video*
v4l2-ctl --list-devices

# Verify X11 forwarding works from inside the container
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  python:3.12-slim bash -c "apt-get install -y x11-apps -q && xclock"

# Check container logs after a crash
docker compose logs --tail=100
```

---

## 12. Troubleshooting

### ❌ `permission denied` connecting to Docker socket

```
permission denied while trying to connect to the Docker daemon socket
```

**Fix:**
```bash
sudo usermod -aG docker $USER
newgrp docker   # apply without logout
```

---

### ❌ `libgl1-mesa-glx` package not found

```
E: Package 'libgl1-mesa-glx' has no installation candidate
```

**Cause:** Debian Trixie renamed the package.
**Fix:** Already handled in the current `Dockerfile` — uses `libgl1` instead.

---

### ❌ `numpy==2.5.1` not found during pip install

```
ERROR: No matching distribution found for numpy==2.5.1
```

**Cause:** numpy 2.5.x requires Python ≥ 3.12 but the Docker image used Python 3.11.
**Fix:** Already handled — Dockerfile uses `python:3.12-slim` to match the conda env.

---

### ❌ OpenCV window does not appear

**Cause:** X11 access not granted to Docker.
**Fix:**
```bash
# Run this on host before starting the container
xhost +local:docker
```

---

### ❌ No audio alerts inside container

**Cause:** PulseAudio socket not reachable.
**Fix:** Verify the socket path exists on the host:
```bash
ls /run/user/1000/pulse/native
# If missing, start PulseAudio:
pulseaudio --start
```

---

### ❌ Webcam not found (`/dev/video0`)

**Cause:** Device index wrong or not passed through.
**Fix:**
```bash
# Find your webcam device
ls /dev/video*
v4l2-ctl --list-devices

# If it's /dev/video2, update docker-compose.yml:
devices:
  - /dev/video2:/dev/video0
```

---

### ❌ RTSP stream not reachable

**Cause:** Container can't reach the camera's IP.
**Fix:** Ensure `network_mode: host` is set in `docker-compose.yml` (it already is by default). Then verify the URL from the host:
```bash
ffprobe rtsp://user:pass@192.168.0.119:554/stream
```

---

*This guide covers the complete Docker lifecycle for the Drowsiness Detection project. For application-level configuration (thresholds, model paths, alert tuning), refer to the main [README.md](README.md).*
