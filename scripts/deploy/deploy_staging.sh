#!/usr/bin/env bash
# Full deploy: local -> S3 -> EC2 (optional) -> health check.
#
# Usage:
#   ./scripts/deploy/deploy_staging.sh                    # S3 artifacts only (no EC2)
#   ./scripts/deploy/deploy_staging.sh --start-ec2        # legacy: git+uv on EC2
#   ./scripts/deploy/deploy_staging.sh --start-ec2 --ecr  # preferred: pull ECR image on EC2
#   ./scripts/deploy/deploy_staging.sh --start-ec2 --ecr --stop-after
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

START_EC2=false
FULL_DATA=false
WITH_CACHE=false
STOP_AFTER=false
NO_PUSH=false
USE_ECR=false

for arg in "$@"; do
  case "$arg" in
    --start-ec2) START_EC2=true ;;
    --full-data) FULL_DATA=true ;;
    --with-cache) WITH_CACHE=true ;;
    --stop-after) STOP_AFTER=true ;;
    --no-push) NO_PUSH=true ;;
    --ecr) USE_ECR=true ;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]

  --start-ec2     Start EC2 and run remote setup (compute charges)
  --ecr           On EC2, pull/run Docker image from ECR (uses GitHub autopush)
  --full-data     Include Data/ in S3 sync
  --with-cache    Include mvp_web/cache/ in S3 sync
  --stop-after    Stop EC2 after successful health check
  --no-push       Skip git push origin staging
EOF
      exit 0
      ;;
    *) die "Unknown option: $arg" ;;
  esac
done

ensure_aws_env_or_bootstrap
require_aws_cli
load_aws_env

if [[ -n "$(cd "$ROOT" && git status --porcelain 2>/dev/null)" ]]; then
  msg "WARNING: uncommitted local changes — deploy may not match git."
fi

BRANCH="${CAPSTONE_BRANCH:-staging}"
if [[ "$NO_PUSH" != true ]]; then
  if (cd "$ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null) | grep -q .; then
    msg "Pushing origin ${BRANCH}..."
    (cd "$ROOT" && git push origin "$BRANCH" 2>/dev/null) || msg "git push skipped (branch may not exist on remote)"
  fi
fi

SYNC_ARGS=()
[[ "$FULL_DATA" == true ]] && SYNC_ARGS+=(--full-data)
[[ "$WITH_CACHE" == true ]] && SYNC_ARGS+=(--with-cache)

msg "=== Deploy: sync local -> S3 ==="
"${SCRIPT_DIR}/sync_local_to_s3.sh" "${SYNC_ARGS[@]}"

if [[ "$START_EC2" != true ]]; then
  msg ""
  msg "S3 sync done. EC2 not started (no compute cost)."
  msg "To deploy to staging: $0 --start-ec2"
  exit 0
fi

if [[ "$USE_ECR" == true ]]; then
  ECR_ARGS=(--start-ec2)
  [[ "$STOP_AFTER" == true ]] && ECR_ARGS+=(--stop-after)
  "${SCRIPT_DIR}/deploy_ecr_to_ec2.sh" "${ECR_ARGS[@]}"
  msg "Deploy complete (ECR -> EC2)."
  exit 0
fi

EBS_ARGS=(--start-ec2 --skip-s3-sync)

# load_to_ebs has its own confirmation prompt (legacy git+uv path)
export CAPSTONE_BRANCH="$BRANCH"
"${SCRIPT_DIR}/load_to_ebs.sh" "${EBS_ARGS[@]}"

if [[ "${EC2_ENABLED:-false}" == true ]]; then
  IP="$(ec2_public_ip)"
  msg "Waiting for /health..."
  for _ in $(seq 1 30); do
    if curl -sf --max-time 5 "http://${IP}:8000/health" >/dev/null 2>&1; then
      msg "Health check OK: http://${IP}:8000/health"
      break
    fi
    sleep 5
  done

  if [[ "$STOP_AFTER" == true ]]; then
    msg "Stopping EC2 (--stop-after)..."
    aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" >/dev/null
    wait_for_ec2_stopped
    msg "EC2 stopped."
  fi
fi

msg "Deploy complete."
