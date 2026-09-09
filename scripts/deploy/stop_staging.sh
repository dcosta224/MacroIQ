#!/usr/bin/env bash
# Stop EC2 staging instance to avoid compute charges.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

load_aws_env
require_aws_cli

if [[ "${EC2_ENABLED:-false}" != "true" || -z "${EC2_INSTANCE_ID:-}" ]]; then
  die "No EC2 configured in deploy/aws.env"
fi

STATE="$(ec2_state)"
if [[ "$STATE" == "stopped" ]]; then
  msg "EC2 ${EC2_INSTANCE_ID} already stopped."
  exit 0
fi

msg "Stopping EC2 ${EC2_INSTANCE_ID}..."
aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$EC2_INSTANCE_ID" >/dev/null
wait_for_ec2_stopped
msg "EC2 stopped. EBS volume still incurs ~\$3/mo storage."
