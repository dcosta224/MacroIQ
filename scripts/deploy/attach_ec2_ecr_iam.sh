#!/usr/bin/env bash
# Attach ECR pull permissions to the existing Capstone EC2 IAM role.
# Safe to re-run. Does not start EC2.
#
# Usage: ./scripts/deploy/attach_ec2_ecr_iam.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

ensure_aws_env_or_bootstrap
require_aws_cli
load_aws_env

: "${AWS_ACCOUNT_ID:?}"
: "${AWS_REGION:?}"
ECR_REPO="${ECR_REPO:-macroiq}"
PROJECT_PREFIX="${PROJECT_PREFIX:-capstone}"
ROLE_NAME="${EC2_IAM_ROLE:-${PROJECT_PREFIX}-ec2-s3-read}"
POLICY_NAME="${PROJECT_PREFIX}-ecr-pull"
TEMPLATE="${ROOT}/infra/aws/iam-ec2-ecr-policy.json"

[[ -f "$TEMPLATE" ]] || die "Missing ${TEMPLATE}"

POLICY_FILE="$(mktemp)"
sed -e "s/AWS_REGION/${AWS_REGION}/g" \
    -e "s/AWS_ACCOUNT_ID/${AWS_ACCOUNT_ID}/g" \
    -e "s/ECR_REPO/${ECR_REPO}/g" \
    "$TEMPLATE" >"$POLICY_FILE"

msg "Attaching ${POLICY_NAME} to role ${ROLE_NAME} (repo ${ECR_REPO})..."
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "$POLICY_NAME" \
  --policy-document "file://${POLICY_FILE}"
rm -f "$POLICY_FILE"

msg "Done. EC2 instance profile can pull ${ECR_REPO} from ECR."
msg "Next (when ready to spend compute): ./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2"
