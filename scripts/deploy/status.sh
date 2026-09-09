#!/usr/bin/env bash
# Show AWS deploy status: S3 manifest, EC2 state, health endpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

if [[ ! -f "$AWS_ENV_FILE" ]]; then
  die "No deploy/aws.env — run ./infra/aws/bootstrap.sh"
fi

load_aws_env
require_aws_cli

msg "=== Capstone AWS status ==="
msg "Region:           ${AWS_REGION}"
msg "Raw bucket:       s3://${S3_BUCKET_RAW}"
msg "Artifacts bucket: s3://${S3_BUCKET_ARTIFACTS}"

TMP="$(mktemp)"
if aws s3 cp "s3://${S3_BUCKET_ARTIFACTS}/deploy-manifest.json" "$TMP" --region "$AWS_REGION" 2>/dev/null; then
  msg "Last deploy manifest:"
  cat "$TMP"
  rm -f "$TMP"
else
  msg "No deploy-manifest.json in artifacts bucket yet."
fi

if [[ "${EC2_ENABLED:-false}" != "true" || -z "${EC2_INSTANCE_ID:-}" ]]; then
  msg ""
  msg "EC2: not configured (S3-only mode)"
  exit 0
fi

STATE="$(ec2_state)"
msg ""
msg "EC2 instance: ${EC2_INSTANCE_ID}"
msg "EC2 state:    ${STATE}"

if [[ "$STATE" == "running" ]]; then
  IP="$(ec2_public_ip)"
  msg "Public IP:    ${IP}"
  msg "Staging URL:  http://${IP}  (map host :80 → container :8000)"
  if curl -sf --max-time 5 "http://${IP}/health" 2>/dev/null; then
    msg "Health:       OK"
  else
    msg "Health:       not responding (app may still be starting)"
  fi
fi
