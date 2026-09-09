#!/usr/bin/env bash
# Upload local datasets and experiment artifacts to S3.
# Does NOT start EC2 — S3 storage cost only.
#
# Usage:
#   ./scripts/deploy/load_to_s3.sh                  # artifacts only
#   ./scripts/deploy/load_to_s3.sh --full-data      # + Data/ (~6.6 GB)
#   ./scripts/deploy/load_to_s3.sh --all            # full-data + cache
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

ARGS=()
ALL=false

for arg in "$@"; do
  case "$arg" in
    --all) ALL=true ;;
    *) ARGS+=("$arg") ;;
  esac
done

if [[ "$ALL" == true ]]; then
  ARGS+=(--full-data --with-cache)
fi

ensure_aws_env_or_bootstrap --s3-only
require_aws_cli
load_aws_env

msg "=== Load local -> S3 (no EC2) ==="
msg "Raw bucket:      s3://${S3_BUCKET_RAW}"
msg "Artifacts bucket: s3://${S3_BUCKET_ARTIFACTS}"
msg ""

exec "${SCRIPT_DIR}/sync_local_to_s3.sh" "${ARGS[@]}"
