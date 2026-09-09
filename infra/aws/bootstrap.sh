#!/usr/bin/env bash
# Interactive one-time AWS setup for Capstone.
# Creates S3 buckets; optionally EC2 staging host.
#
# Usage:
#   ./infra/aws/bootstrap.sh              # interactive (asks about EC2)
#   ./infra/aws/bootstrap.sh --s3-only    # S3 buckets only — no EC2, no compute cost
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="${ROOT}/deploy"
AWS_ENV_FILE="${DEPLOY_DIR}/aws.env"
INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S3_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --s3-only) S3_ONLY=true ;;
    -h|--help)
      echo "Usage: $0 [--s3-only]"
      exit 0
      ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

msg() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

prompt() {
  local var_name="$1"
  local prompt_text="$2"
  local default="$3"
  local value
  read -r -p "${prompt_text} [${default}]: " value
  value="${value:-$default}"
  printf -v "$var_name" '%s' "$value"
}

prompt_yes_no() {
  local prompt_text="$1"
  local default="${2:-Y}"
  local answer
  if [[ "$default" == "Y" ]]; then
    read -r -p "${prompt_text} [Y/n]: " answer
    answer="${answer:-Y}"
  else
    read -r -p "${prompt_text} [y/N]: " answer
    answer="${answer:-N}"
  fi
  [[ "$answer" =~ ^[Yy] ]]
}

command -v aws >/dev/null 2>&1 || die "Install AWS CLI: https://docs.aws.amazon.com/cli/"
aws sts get-caller-identity >/dev/null 2>&1 || die "Not logged in. Run: aws login"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
msg ""
msg "=== Capstone AWS bootstrap ==="
msg "Account: ${ACCOUNT_ID}"
msg ""

DEFAULT_REGION="$(aws configure get region 2>/dev/null || echo us-east-1)"
prompt AWS_REGION "AWS region" "$DEFAULT_REGION"
prompt PROJECT_PREFIX "Project prefix (bucket names)" "capstone"

MY_IP=""
if command -v curl >/dev/null 2>&1; then
  MY_IP="$(curl -sf --max-time 5 https://ifconfig.me 2>/dev/null || true)"
fi
prompt MY_CIDR "Your IP for SSH/API access (CIDR)" "${MY_IP:+${MY_IP}/32}"
[[ -n "$MY_CIDR" ]] || die "Need your public IP CIDR for security group (e.g. 1.2.3.4/32)"

prompt GIT_REPO "Git repository URL" "https://github.com/dcosta224/Capstone.git"
prompt GIT_BRANCH "Deploy branch" "staging"

S3_BUCKET_RAW="${PROJECT_PREFIX}-${ACCOUNT_ID}-raw"
S3_BUCKET_ARTIFACTS="${PROJECT_PREFIX}-${ACCOUNT_ID}-artifacts"

msg ""
msg "Will create buckets:"
msg "  s3://${S3_BUCKET_RAW}"
msg "  s3://${S3_BUCKET_ARTIFACTS}"

CREATE_EC2=false
if [[ "$S3_ONLY" == true ]]; then
  msg ""
  msg "--s3-only: skipping EC2 (no compute/EBS cost beyond S3 storage)."
else
  msg ""
  if prompt_yes_no "Create EC2 staging instance? (EBS ~\$3/mo even when stopped)" "N"; then
    CREATE_EC2=true
  fi
fi

INSTANCE_TYPE="t3.small"
KEY_NAME="${PROJECT_PREFIX}-staging"
KEY_PATH="${HOME}/.ssh/${KEY_NAME}.pem"
EC2_INSTANCE_ID=""
EC2_ENABLED=false

# --- S3 buckets ---
for bucket in "$S3_BUCKET_RAW" "$S3_BUCKET_ARTIFACTS"; do
  if aws s3api head-bucket --bucket "$bucket" --region "$AWS_REGION" 2>/dev/null; then
    msg "Bucket exists: ${bucket}"
  else
    msg "Creating bucket: ${bucket}"
    if [[ "$AWS_REGION" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "$bucket" --region "$AWS_REGION"
    else
      aws s3api create-bucket --bucket "$bucket" --region "$AWS_REGION" \
        --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi
  fi
done

aws s3api put-bucket-versioning \
  --bucket "$S3_BUCKET_ARTIFACTS" \
  --versioning-configuration Status=Enabled \
  --region "$AWS_REGION" 2>/dev/null || true

# --- EC2 (optional) ---
if [[ "$CREATE_EC2" == true ]]; then
  SG_NAME="${PROJECT_PREFIX}-staging-sg"
  SG_ID="$(aws ec2 describe-security-groups --region "$AWS_REGION" \
    --filters "Name=group-name,Values=${SG_NAME}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")"

  if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    msg "Creating security group: ${SG_NAME}"
    SG_ID="$(aws ec2 create-security-group \
      --region "$AWS_REGION" \
      --group-name "$SG_NAME" \
      --description "Capstone staging SSH + MVP API" \
      --query GroupId --output text)"
    aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" \
      --protocol tcp --port 22 --cidr "$MY_CIDR"
    aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" \
      --protocol tcp --port 8000 --cidr "$MY_CIDR"
  else
    msg "Security group exists: ${SG_ID}"
  fi

  if ! aws ec2 describe-key-pairs --region "$AWS_REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
    msg "Creating key pair: ${KEY_NAME} -> ${KEY_PATH}"
    aws ec2 create-key-pair --region "$AWS_REGION" --key-name "$KEY_NAME" \
      --query KeyMaterial --output text >"$KEY_PATH"
    chmod 600 "$KEY_PATH"
  else
    msg "Key pair exists: ${KEY_NAME}"
    [[ -f "$KEY_PATH" ]] || msg "WARNING: ${KEY_PATH} not found locally — use existing key or delete key pair in console."
  fi

  if prompt_yes_no "Instance type OK as t3.small? (n = choose t3.medium)" "Y"; then
    :
  else
    INSTANCE_TYPE="t3.medium"
  fi

  AMI_ID="$(aws ssm get-parameters --region "$AWS_REGION" \
    --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --query 'Parameters[0].Value' --output text)"

  ROLE_NAME="${PROJECT_PREFIX}-ec2-s3-read"
  PROFILE_NAME="${PROJECT_PREFIX}-ec2-instance-profile"

  if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    msg "Creating IAM role: ${ROLE_NAME}"
    aws iam create-role --role-name "$ROLE_NAME" \
      --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
          "Effect": "Allow",
          "Principal": {"Service": "ec2.amazonaws.com"},
          "Action": "sts:AssumeRole"
        }]
      }' >/dev/null
    POLICY_FILE="$(mktemp)"
    sed -e "s/BUCKET_RAW/${S3_BUCKET_RAW}/g" -e "s/BUCKET_ARTIFACTS/${S3_BUCKET_ARTIFACTS}/g" \
      "${INFRA_DIR}/iam-ec2-s3-policy.json" >"$POLICY_FILE"
    aws iam put-role-policy --role-name "$ROLE_NAME" \
      --policy-name "${PROJECT_PREFIX}-s3-read" \
      --policy-document "file://${POLICY_FILE}"
    rm -f "$POLICY_FILE"
    ECR_POLICY_FILE="$(mktemp)"
    sed -e "s/AWS_REGION/${AWS_REGION}/g" \
        -e "s/AWS_ACCOUNT_ID/${ACCOUNT_ID}/g" \
        -e "s/ECR_REPO/macroiq/g" \
      "${INFRA_DIR}/iam-ec2-ecr-policy.json" >"$ECR_POLICY_FILE"
    aws iam put-role-policy --role-name "$ROLE_NAME" \
      --policy-name "${PROJECT_PREFIX}-ecr-pull" \
      --policy-document "file://${ECR_POLICY_FILE}"
    rm -f "$ECR_POLICY_FILE"
  fi

  if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
    msg "Creating instance profile: ${PROFILE_NAME}"
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
    aws iam add-role-to-instance-profile \
      --instance-profile-name "$PROFILE_NAME" \
      --role-name "$ROLE_NAME"
    sleep 10
  fi

  EXISTING="$(aws ec2 describe-instances --region "$AWS_REGION" \
    --filters "Name=tag:Name,Values=${PROJECT_PREFIX}-staging" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")"

  if [[ "$EXISTING" != "None" && -n "$EXISTING" ]]; then
    EC2_INSTANCE_ID="$EXISTING"
    msg "EC2 already exists: ${EC2_INSTANCE_ID}"
  else
    msg "Launching EC2 (${INSTANCE_TYPE})..."
    EC2_INSTANCE_ID="$(aws ec2 run-instances --region "$AWS_REGION" \
      --image-id "$AMI_ID" \
      --instance-type "$INSTANCE_TYPE" \
      --key-name "$KEY_NAME" \
      --security-group-ids "$SG_ID" \
      --iam-instance-profile "Name=${PROFILE_NAME}" \
      --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PROJECT_PREFIX}-staging}]" \
      --query 'Instances[0].InstanceId' --output text)"
    msg "Launched: ${EC2_INSTANCE_ID} (stopping to avoid compute charges until you deploy)"
    aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" >/dev/null
    aws ec2 wait instance-stopped --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID"
  fi
  EC2_ENABLED=true
fi

mkdir -p "$DEPLOY_DIR"
cat >"$AWS_ENV_FILE" <<EOF
# Generated by infra/aws/bootstrap.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
AWS_REGION=${AWS_REGION}
AWS_ACCOUNT_ID=${ACCOUNT_ID}

S3_BUCKET_RAW=${S3_BUCKET_RAW}
S3_BUCKET_ARTIFACTS=${S3_BUCKET_ARTIFACTS}

EC2_ENABLED=${EC2_ENABLED}
EC2_INSTANCE_ID=${EC2_INSTANCE_ID}
EC2_KEY_PATH=${KEY_PATH}
EC2_SSH_USER=ec2-user

CAPSTONE_GIT_REPO=${GIT_REPO}
CAPSTONE_BRANCH=${GIT_BRANCH}
CAPSTONE_ROOT=/opt/capstone

ECR_REPO=macroiq
ECR_IMAGE_TAG=deployment
EOF

msg ""
msg "=== Bootstrap complete ==="
msg "Wrote: ${AWS_ENV_FILE}"
msg ""
msg "Next steps (S3 only — no EC2 cost):"
msg "  ./scripts/deploy/load_to_s3.sh --all"
msg ""
if [[ "$EC2_ENABLED" == true ]]; then
  msg "When ready for staging (starts EC2 — compute charges apply):"
  msg "  ./scripts/deploy/load_to_ebs.sh --start-ec2"
  msg "  scp -i ${KEY_PATH} .env ec2-user@<ip>:/opt/capstone/.env"
  msg ""
  msg "Stop EC2 when done:"
  msg "  ./scripts/deploy/stop_staging.sh"
fi
