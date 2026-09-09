# AWS workflow (Capstone)

Local dev on your Mac; S3 for durable files; optional on-demand EC2 for staging demos.
Supabase stays your shared database.

**Container path (GitHub → ECR → EC2):** see **[ECR_EC2_DEPLOY.md](ECR_EC2_DEPLOY.md)** for the full write-up of automated image push and Docker-on-EC2 hosting.

## Cost cheat sheet

| Action | EC2 runs? | Typical cost |
|--------|-----------|--------------|
| `bootstrap.sh --s3-only` | No | S3 storage ~$0.20/mo per 7 GB |
| `load_to_s3.sh` | No | S3 only |
| `pull_artifacts.sh` | No | S3 requests only |
| `bootstrap.sh` + EC2 | Creates instance **stopped** | EBS ~$3/mo |
| `load_to_ebs.sh --start-ec2` | **Yes** | + ~$0.02/hr t3.small |
| `stop_staging.sh` | Stops EC2 | Compute stops; EBS remains |

## Daniel — first-time setup

```bash
cd Capstone

# 1. AWS login (if needed)
aws login

# 2. S3 buckets only — no EC2, no compute bill
chmod +x infra/aws/bootstrap.sh scripts/deploy/*.sh
./infra/aws/bootstrap.sh --s3-only

# 3. Upload data + artifacts to S3 (~6.6 GB first time)
./scripts/deploy/load_to_s3.sh --all

# 4. (Optional) Create EC2 when ready for shared staging URL
./infra/aws/bootstrap.sh   # answer Y to EC2; instance launches then stops

# 5. (Optional) Deploy to EC2 — requires --start-ec2
./scripts/deploy/load_to_ebs.sh --start-ec2

# 6. Copy secrets once
scp -i ~/.ssh/capstone-staging.pem .env ec2-user@<ip>:/opt/capstone/.env
./scripts/deploy/load_to_ebs.sh --start-ec2   # restart after .env

# 7. Stop when done
./scripts/deploy/stop_staging.sh
```

## Daniel — day-to-day deploy

```bash
# After local testing — S3 only (safe, no EC2)
./scripts/deploy/deploy_staging.sh

# Full staging demo (starts EC2) — legacy git+uv on the box
./scripts/deploy/deploy_staging.sh --start-ec2 --with-cache

# Preferred: run the GitHub/ECR Docker image on EC2 (autopush -> pull -> restart)
# Requires EC2 instance + IAM ECR pull (once): ./scripts/deploy/attach_ec2_ecr_iam.sh
./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2
# or: ./scripts/deploy/deploy_staging.sh --start-ec2 --ecr --stop-after

# Check status
./scripts/deploy/status.sh
```

## ECR → EC2 (uses GitHub Actions image)

GitHub Actions on `deployment` pushes `macroiq:deployment` to ECR. EC2 does **not** auto-pull on every push (avoids starting compute). When you want a demo:

```bash
# Once per AWS account/role (if EC2 was created before ECR wiring):
./scripts/deploy/attach_ec2_ecr_iam.sh

# Copy .env to the instance once (Supabase PG_* etc.):
# scp -i ~/.ssh/capstone-staging.pem .env ec2-user@<ip>:/opt/capstone/.env

# On for demo — starts EC2, docker pull, restart container
./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2

# Off — stop compute
./scripts/deploy/stop_staging.sh
```

## Partner — three commands

```bash
# 1. Get bucket names from Daniel; copy template:
cp deploy/aws.env.example deploy/aws.env
# Edit S3_BUCKET_RAW and S3_BUCKET_ARTIFACTS

# 2. Pull latest experiment files (no EC2)
aws login
./scripts/deploy/pull_artifacts.sh

# 3. Use staging demo (after Daniel deploys)
./scripts/deploy/status.sh
# open http://<ip>:8000
```

## What lives where

| Location | Contents |
|----------|----------|
| **Git** | Code, notebooks tracked in repo, SQL |
| **S3 raw** | `Data/` (USDA, RecipeNLG) |
| **S3 artifacts** | `scratch/EDA/*.parquet`, feasibility reports, deploy manifest |
| **EC2 EBS** | Docker runtime + `.env`; pulls `macroiq` from ECR (legacy path: git clone + `.venv`) |
| **Supabase** | `usda`, `recipe`, `resolved_recipes`, embeddings |
| **Never S3** | `.env`, secrets |

## Safety flags

- `--s3-only` on bootstrap → buckets only
- `load_to_ebs.sh` requires `--start-ec2` or it exits
- `deploy_staging.sh` without `--start-ec2` → S3 sync only
