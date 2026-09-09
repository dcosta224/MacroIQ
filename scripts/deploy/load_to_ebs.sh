#!/usr/bin/env bash
# Hydrate EC2 EBS from S3 and run remote_setup.sh.
# REQUIRES --start-ec2 (starts instance — compute charges apply).
#
# Usage:
#   ./scripts/deploy/load_to_ebs.sh --start-ec2
#   ./scripts/deploy/load_to_ebs.sh --start-ec2 --full-data   # sync local Data/ to S3 first
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

START_EC2=false
SYNC_FULL_DATA=false
SKIP_S3_SYNC=false

for arg in "$@"; do
  case "$arg" in
    --start-ec2) START_EC2=true ;;
    --full-data) SYNC_FULL_DATA=true ;;
    --skip-s3-sync) SKIP_S3_SYNC=true ;;
    -h|--help)
      echo "Usage: $0 --start-ec2 [--full-data]"
      echo "  --start-ec2   Required. Starts EC2 and runs remote setup."
      exit 0
      ;;
    *) die "Unknown option: $arg (EC2 ops require --start-ec2)" ;;
  esac
done

if [[ "$START_EC2" != true ]]; then
  die "Refusing to touch EC2 without --start-ec2. Use load_to_s3.sh for S3-only uploads."
fi

ensure_aws_env_or_bootstrap
require_aws_cli
load_aws_env

if [[ "${EC2_ENABLED:-false}" != "true" || -z "${EC2_INSTANCE_ID:-}" ]]; then
  die "No EC2 in deploy/aws.env. Re-run: ./infra/aws/bootstrap.sh (without --s3-only)"
fi

msg "=== Load S3 -> EC2 EBS ==="
msg "This will START EC2 instance ${EC2_INSTANCE_ID} (compute charges apply)."

if ! prompt_yes_no "Continue?" "N"; then
  die "Aborted."
fi

if [[ "$SKIP_S3_SYNC" != true ]]; then
  if [[ "$SYNC_FULL_DATA" == true ]]; then
    "${SCRIPT_DIR}/sync_local_to_s3.sh" --full-data
  else
    "${SCRIPT_DIR}/sync_local_to_s3.sh"
  fi
fi

start_ec2_if_needed

msg "Running remote_setup.sh on EC2..."
ssh_ec2 env \
  S3_BUCKET_RAW="${S3_BUCKET_RAW}" \
  S3_BUCKET_ARTIFACTS="${S3_BUCKET_ARTIFACTS}" \
  AWS_REGION="${AWS_REGION}" \
  CAPSTONE_GIT_REPO="${CAPSTONE_GIT_REPO}" \
  CAPSTONE_BRANCH="${CAPSTONE_BRANCH}" \
  CAPSTONE_ROOT="${CAPSTONE_ROOT:-/opt/capstone}" \
  bash -s < "${SCRIPT_DIR}/remote_setup.sh"

IP="$(ec2_public_ip)"
msg ""
msg "=== EC2 ready ==="
msg "Staging URL: http://${IP}:8000"
msg "Health:      curl http://${IP}:8000/health"
msg ""
msg "Copy secrets if not done:"
msg "  scp -i ${EC2_KEY_PATH} .env ${EC2_SSH_USER}@${IP}:${CAPSTONE_ROOT:-/opt/capstone}/.env"
msg "  ./scripts/deploy/load_to_ebs.sh --start-ec2   # restart after .env"
msg ""
msg "Stop when done: ./scripts/deploy/stop_staging.sh"
