#!/usr/bin/env bash
# Runs ON the EC2 instance: hydrate EBS from S3, pull code, install deps, restart app.
# Invoked via SSH from load_to_ebs.sh — do not run locally.
set -euo pipefail

: "${CAPSTONE_ROOT:=/opt/capstone}"
: "${S3_BUCKET_RAW:?}"
: "${S3_BUCKET_ARTIFACTS:?}"
: "${AWS_REGION:?}"
: "${CAPSTONE_GIT_REPO:?}"
: "${CAPSTONE_BRANCH:=deployment}"

log() { printf '[remote_setup] %s\n' "$*"; }

if ! command -v aws >/dev/null 2>&1; then
  log "Installing AWS CLI..."
  sudo dnf install -y aws-cli 2>/dev/null || sudo yum install -y aws-cli
fi

if ! command -v git >/dev/null 2>&1; then
  sudo dnf install -y git
fi

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

sudo mkdir -p "$CAPSTONE_ROOT"
sudo chown -R ec2-user:ec2-user "$CAPSTONE_ROOT"

if [[ ! -d "${CAPSTONE_ROOT}/.git" ]]; then
  log "Cloning ${CAPSTONE_GIT_REPO} -> ${CAPSTONE_ROOT}"
  git clone "$CAPSTONE_GIT_REPO" "$CAPSTONE_ROOT"
fi

cd "$CAPSTONE_ROOT"
git fetch origin
git checkout "$CAPSTONE_BRANCH"
git pull origin "$CAPSTONE_BRANCH"

log "Syncing S3 raw Data/ -> ${CAPSTONE_ROOT}/Data/"
aws s3 sync "s3://${S3_BUCKET_RAW}/Data/" "${CAPSTONE_ROOT}/Data/" --region "$AWS_REGION" || true

log "Syncing S3 artifacts scratch/ -> ${CAPSTONE_ROOT}/scratch/"
mkdir -p "${CAPSTONE_ROOT}/scratch"
aws s3 sync "s3://${S3_BUCKET_ARTIFACTS}/scratch/" "${CAPSTONE_ROOT}/scratch/" --region "$AWS_REGION" || true

log "uv sync (agent runtime deps)..."
export PATH="${HOME}/.local/bin:${PATH}"
cd "$CAPSTONE_ROOT"
uv sync --no-dev

log "Installing systemd unit..."
sudo cp "${CAPSTONE_ROOT}/infra/aws/capstone-mvp.service" /etc/systemd/system/capstone-mvp.service
sudo systemctl daemon-reload
sudo systemctl enable capstone-mvp

if [[ -f "${CAPSTONE_ROOT}/.env" ]]; then
  sudo systemctl restart capstone-mvp
  log "Restarted capstone-mvp service (recipe_opt_web)."
else
  log "Skipping service start — no .env at ${CAPSTONE_ROOT}/.env"
fi

log "Disk usage:"
df -h / | tail -1
du -sh "${CAPSTONE_ROOT}/Data" 2>/dev/null || true
du -sh "${CAPSTONE_ROOT}/scratch" 2>/dev/null || true
log "remote_setup complete."
