# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder
#   Installs all Python dependencies into /install so the final image
#   only copies compiled wheels without build-time toolchains.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System packages needed only at build time (compilers, headers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install Python deps into a separate prefix so they're easy to copy over
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime
#   Lean image that only carries the runtime system libs and the pre-built
#   Python wheels from the builder stage.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="VishnuAlgosium" \
      description="Drowsiness & driver behaviour detection" \
      org.opencontainers.image.source="https://github.com/VishnuAlgosium/drowsiness_detection"

# ── Runtime system libraries ──────────────────────────────────────────────────
#   libgl1            – OpenCV needs libGL at runtime
#   libglib2.0-0      – GLib (MediaPipe / GIO)
#   libsm6 libxext6   – X11 extension libs (OpenCV imshow)
#   libxrender1       – X rendering (OpenCV window)
#   libgomp1          – OpenMP (NCNN multi-threaded inference)
#   libsdl2-2.0-0     – SDL2 audio backend for pygame
#   pulseaudio-utils  – pactl / pacmd; lets the container speak to the host PA socket
#   libgstreamer1.0-0 – needed by some OpenCV builds for camera back-ends
#   v4l-utils         – optional but useful for debugging USB webcam device nodes
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libsdl2-2.0-0 \
        pulseaudio-utils \
        libgstreamer1.0-0 \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Python packages from the builder
COPY --from=builder /install /usr/local

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app

COPY main.py        ./
COPY src/           ./src/
COPY models/        ./models/

# Log and employee directories are expected to be mounted as volumes.
# Create them so the app can write even without a mount.
RUN mkdir -p /app/logs /app/employee

# ── Non-root user ─────────────────────────────────────────────────────────────
# Running as root inside a container is a security risk.
# The 'video' group (GID 44) allows access to /dev/video* on most Linux hosts.
RUN groupadd -r drowsy \
    && useradd -r -g drowsy -G video,audio drowsy \
    && chown -R drowsy:drowsy /app

USER drowsy

# ── Environment defaults ──────────────────────────────────────────────────────
# RTSP_URL   – leave empty to use the USB webcam (CAMERA_INDEX=0)
# DISPLAY    – set to :0 or passed in from the host for X11 forwarding
# PULSE_SERVER – PulseAudio server socket; override at runtime if needed
ENV RTSP_URL="" \
    DISPLAY=":0" \
    PULSE_SERVER="unix:/run/user/1000/pulse/native" \
    PYTHONUNBUFFERED=1

# ── Entry point ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]
