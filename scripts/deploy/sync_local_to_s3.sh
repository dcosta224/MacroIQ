#!/usr/bin/env bash
# Low-level: sync local Capstone paths to S3 buckets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

FULL_DATA=false
WITH_CACHE=false
INCLUDE_CSV=false
INCLUDE_MLRUNS=false

usage() {
  cat <<EOF
Usage: $0 [options]

  --full-data      Sync Data/ to raw bucket (~6.6 GB)
  --with-cache     Sync mvp_web/cache/ to artifacts bucket
  --include-csv    Include scratch/**/*.csv in artifacts sync
  --include-mlruns Sync mlruns/ to artifacts bucket
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-data) FULL_DATA=true ;;
    --with-cache) WITH_CACHE=true ;;
    --include-csv) INCLUDE_CSV=true ;;
    --include-mlruns) INCLUDE_MLRUNS=true ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

load_aws_env
require_aws_cli

sync_excludes() {
  local extra=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [[ -n "$line" ]] && extra+=(--exclude "$line")
  done < "${DEPLOY_DIR}/s3-excludes"
  aws s3 sync "$@" "${extra[@]}"
}

if [[ "$FULL_DATA" == true ]]; then
  if [[ ! -d "${ROOT}/Data" ]]; then
    die "Data/ not found at ${ROOT}/Data"
  fi
  msg "Syncing Data/ -> s3://${S3_BUCKET_RAW}/Data/"
  sync_excludes "${ROOT}/Data/" "s3://${S3_BUCKET_RAW}/Data/" --region "$AWS_REGION"
fi

msg "Syncing scratch/EDA artifacts -> s3://${S3_BUCKET_ARTIFACTS}/scratch/EDA/"
if [[ -d "${ROOT}/scratch/EDA" ]]; then
  aws s3 sync "${ROOT}/scratch/EDA/" "s3://${S3_BUCKET_ARTIFACTS}/scratch/EDA/" \
    --region "$AWS_REGION" \
    --exclude "*" \
    --include "*.parquet" \
    --include "*.pkl" \
    --include "feasibility_report.json" \
    --include "run_manifest.json" \
    --include "run.log"
  if [[ "$INCLUDE_CSV" == true ]]; then
    aws s3 sync "${ROOT}/scratch/EDA/" "s3://${S3_BUCKET_ARTIFACTS}/scratch/EDA/" \
      --region "$AWS_REGION" \
      --exclude "*" \
      --include "*.csv"
  fi
else
  msg "No scratch/EDA/ — skipping artifact sync."
fi

if [[ "$WITH_CACHE" == true && -d "${ROOT}/mvp_web/cache" ]]; then
  msg "Syncing mvp_web/cache/ -> artifacts bucket"
  sync_excludes "${ROOT}/mvp_web/cache/" "s3://${S3_BUCKET_ARTIFACTS}/mvp_web/cache/" \
    --region "$AWS_REGION"
fi

if [[ "$INCLUDE_MLRUNS" == true && -d "${ROOT}/mlruns" ]]; then
  msg "Syncing mlruns/ -> artifacts bucket"
  sync_excludes "${ROOT}/mlruns/" "s3://${S3_BUCKET_ARTIFACTS}/mlruns/" \
    --region "$AWS_REGION"
fi

write_deploy_manifest "$FULL_DATA" "$WITH_CACHE"
msg "S3 sync complete."
