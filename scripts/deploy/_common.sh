#!/usr/bin/env bash
# Shared helpers for Capstone AWS deploy scripts.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../deploy" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AWS_ENV_FILE="${DEPLOY_DIR}/aws.env"

msg() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

load_aws_env() {
  if [[ ! -f "$AWS_ENV_FILE" ]]; then
    die "Missing ${AWS_ENV_FILE}. Run: ./infra/aws/bootstrap.sh"
  fi
  # shellcheck disable=SC1090
  source "$AWS_ENV_FILE"
  : "${AWS_REGION:?AWS_REGION not set in aws.env}"
  : "${S3_BUCKET_RAW:?S3_BUCKET_RAW not set in aws.env}"
  : "${S3_BUCKET_ARTIFACTS:?S3_BUCKET_ARTIFACTS not set in aws.env}"
}

require_aws_cli() {
  require_cmd aws
  aws sts get-caller-identity >/dev/null 2>&1 || die "AWS CLI not authenticated. Run: aws login"
}

git_short_sha() {
  (cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null) || echo "unknown"
}

write_deploy_manifest() {
  local full_data="${1:-false}"
  local with_cache="${2:-false}"
  local tmp
  tmp="$(mktemp)"
  cat >"$tmp" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_sha": "$(git_short_sha)",
  "git_branch": "$(cd "$ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)",
  "full_data": ${full_data},
  "with_cache": ${with_cache},
  "s3_bucket_raw": "${S3_BUCKET_RAW}",
  "s3_bucket_artifacts": "${S3_BUCKET_ARTIFACTS}"
}
EOF
  aws s3 cp "$tmp" "s3://${S3_BUCKET_ARTIFACTS}/deploy-manifest.json" --region "$AWS_REGION"
  rm -f "$tmp"
  msg "Wrote s3://${S3_BUCKET_ARTIFACTS}/deploy-manifest.json"
}

ec2_state() {
  load_aws_env
  [[ -n "${EC2_INSTANCE_ID:-}" ]] || die "EC2_INSTANCE_ID not set in aws.env (bootstrap with EC2 or set manually)"
  aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids "$EC2_INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null || echo "unknown"
}

ec2_public_ip() {
  load_aws_env
  aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids "$EC2_INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text
}

wait_for_ec2_running() {
  load_aws_env
  msg "Waiting for EC2 ${EC2_INSTANCE_ID} to reach running..."
  aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID"
  sleep 20
}

wait_for_ec2_stopped() {
  load_aws_env
  msg "Waiting for EC2 ${EC2_INSTANCE_ID} to stop..."
  aws ec2 wait instance-stopped --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID"
}

start_ec2_if_needed() {
  load_aws_env
  local state
  state="$(ec2_state)"
  case "$state" in
    running) msg "EC2 already running." ;;
    stopped)
      msg "Starting EC2 ${EC2_INSTANCE_ID}..."
      aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" >/dev/null
      wait_for_ec2_running
      ;;
    *) die "EC2 instance is in state: ${state}" ;;
  esac
}

ssh_ec2() {
  load_aws_env
  : "${EC2_SSH_USER:=ec2-user}"
  : "${EC2_KEY_PATH:?EC2_KEY_PATH not set in aws.env}"
  [[ -f "$EC2_KEY_PATH" ]] || die "SSH key not found: ${EC2_KEY_PATH}"
  local ip
  ip="$(ec2_public_ip)"
  [[ "$ip" != "None" && -n "$ip" ]] || die "EC2 has no public IP"
  ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
    "${EC2_SSH_USER}@${ip}" "$@"
}

prompt_yes_no() {
  local prompt="$1"
  local default="${2:-Y}"
  local answer
  if [[ "$default" == "Y" ]]; then
    read -r -p "${prompt} [Y/n]: " answer
    answer="${answer:-Y}"
  else
    read -r -p "${prompt} [y/N]: " answer
    answer="${answer:-N}"
  fi
  [[ "$answer" =~ ^[Yy] ]]
}

ensure_aws_env_or_bootstrap() {
  if [[ ! -f "$AWS_ENV_FILE" ]]; then
    msg "No deploy/aws.env found."
    if prompt_yes_no "Run interactive bootstrap now?" "Y"; then
      "${ROOT}/infra/aws/bootstrap.sh" "$@"
      return
    fi
    die "Create deploy/aws.env first: ./infra/aws/bootstrap.sh"
  fi
}
