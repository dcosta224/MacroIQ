#!/usr/bin/env bash
# Start EC2 (optional), pull latest ECR image, restart Docker MVP service.
# Cost: EC2 compute only while the instance is running — stop when done.
#
# Usage:
#   ./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2
#   ./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2 --stop-after
#   ECR_IMAGE_TAG=deployment ./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

START_EC2=false
STOP_AFTER=false

for arg in "$@"; do
  case "$arg" in
    --start-ec2) START_EC2=true ;;
    --stop-after) STOP_AFTER=true ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) die "Unknown option: $arg" ;;
  esac
done

if [[ "$START_EC2" != true ]]; then
  die "Refusing to touch EC2 without --start-ec2 (avoids accidental compute charges)."
fi

ensure_aws_env_or_bootstrap
require_aws_cli
load_aws_env

if [[ "${EC2_ENABLED:-false}" != "true" || -z "${EC2_INSTANCE_ID:-}" ]]; then
  die "No EC2 in deploy/aws.env. Ask your partner to run: ./infra/aws/bootstrap.sh (with EC2), then set EC2_ENABLED=true"
fi

: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID not set in aws.env}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-macroiq}"
ECR_IMAGE_TAG="${ECR_IMAGE_TAG:-deployment}"
ECR_IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${ECR_IMAGE_TAG}"
CAPSTONE_ROOT="${CAPSTONE_ROOT:-/opt/capstone}"

msg "=== Deploy ECR image to EC2 ==="
msg "Instance:  ${EC2_INSTANCE_ID}"
msg "Image:     ${ECR_IMAGE_URI}"
msg "This STARTS EC2 if stopped (compute charges apply)."

if ! prompt_yes_no "Continue?" "N"; then
  die "Aborted."
fi

start_ec2_if_needed

UNIT_LOCAL="${ROOT}/infra/aws/capstone-mvp-docker.service"
[[ -f "$UNIT_LOCAL" ]] || die "Missing ${UNIT_LOCAL}"

msg "Uploading systemd unit + remote setup..."
: "${EC2_SSH_USER:=ec2-user}"
: "${EC2_KEY_PATH:?EC2_KEY_PATH not set}"
IP="$(ec2_public_ip)"
scp -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=accept-new \
  "$UNIT_LOCAL" \
  "${SCRIPT_DIR}/remote_setup_ecr.sh" \
  "${EC2_SSH_USER}@${IP}:/tmp/"

ssh_ec2 env \
  AWS_REGION="${AWS_REGION}" \
  ECR_IMAGE_URI="${ECR_IMAGE_URI}" \
  CAPSTONE_ROOT="${CAPSTONE_ROOT}" \
  bash /tmp/remote_setup_ecr.sh

msg "Waiting for /health (image may take a few minutes on first pull)..."
OK=false
for _ in $(seq 1 60); do
  if curl -sf --max-time 5 "http://${IP}/health" >/dev/null 2>&1; then
    msg "Health check OK: http://${IP}/health"
    OK=true
    break
  fi
  sleep 5
done
[[ "$OK" == true ]] || msg "WARNING: /health not ready yet — check: ssh and journalctl -u capstone-mvp-docker"

msg ""
msg "Staging URL: http://${IP}  (or http://macroiq.org if DNS A @ points here)"
msg "If first time, copy secrets:"
msg "  scp -i ${EC2_KEY_PATH} .env ${EC2_SSH_USER}@${IP}:${CAPSTONE_ROOT}/.env"
msg "  $0 --start-ec2"

if [[ "$STOP_AFTER" == true ]]; then
  msg "Stopping EC2 (--stop-after)..."
  aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" >/dev/null
  wait_for_ec2_stopped
  msg "EC2 stopped."
fi

msg "Done."
