#!/usr/bin/env bash
# Runs ON the EC2 instance: install Docker if needed, pull agent image from ECR, restart systemd unit.
# Invoked via SSH from deploy_ecr_to_ec2.sh — do not run locally.
set -euo pipefail

: "${AWS_REGION:?}"
: "${ECR_IMAGE_URI:?}"
: "${CAPSTONE_ROOT:=/opt/capstone}"
: "${SERVICE_UNIT_URL:=}"  # optional: raw content passed via stdin as unit file instead

log() { printf '[remote_setup_ecr] %s\n' "$*"; }

if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker..."
  sudo dnf install -y docker 2>/dev/null || sudo yum install -y docker
  sudo systemctl enable --now docker
  sudo usermod -aG docker ec2-user || true
fi

if ! command -v aws >/dev/null 2>&1; then
  log "Installing AWS CLI..."
  sudo dnf install -y aws-cli 2>/dev/null || sudo yum install -y aws-cli
fi

sudo mkdir -p "$CAPSTONE_ROOT"
sudo chown -R ec2-user:ec2-user "$CAPSTONE_ROOT" 2>/dev/null || true

log "Writing /etc/capstone-ecr.env (image ${ECR_IMAGE_URI})"
printf 'ECR_IMAGE_URI=%s\nAWS_REGION=%s\n' "$ECR_IMAGE_URI" "$AWS_REGION" | sudo tee /etc/capstone-ecr.env >/dev/null

# Unit file is uploaded by the caller to /tmp/capstone-mvp-docker.service
if [[ -f /tmp/capstone-mvp-docker.service ]]; then
  sudo cp /tmp/capstone-mvp-docker.service /etc/systemd/system/capstone-mvp-docker.service
else
  die_msg="Missing /tmp/capstone-mvp-docker.service — deploy script should scp the unit file."
  log "ERROR: ${die_msg}"
  exit 1
fi

log "ECR login..."
aws ecr get-login-password --region "$AWS_REGION" | \
  sudo docker login --username AWS --password-stdin \
  "$(echo "$ECR_IMAGE_URI" | cut -d/ -f1)"

log "Pulling ${ECR_IMAGE_URI}..."
sudo docker pull "$ECR_IMAGE_URI"

# Prefer Docker unit; stop legacy uv-based service if present (port 8000 clash).
if systemctl list-unit-files | grep -q '^capstone-mvp.service'; then
  log "Disabling legacy capstone-mvp (uv) service..."
  sudo systemctl disable --now capstone-mvp 2>/dev/null || true
fi

sudo systemctl daemon-reload
sudo systemctl enable capstone-mvp-docker

if [[ -f "${CAPSTONE_ROOT}/.env" ]]; then
  sudo systemctl restart capstone-mvp-docker
  log "Restarted capstone-mvp-docker."
else
  log "WARNING: no ${CAPSTONE_ROOT}/.env — container not started."
  log "Copy .env then re-run deploy_ecr_to_ec2.sh"
fi

log "remote_setup_ecr complete."
