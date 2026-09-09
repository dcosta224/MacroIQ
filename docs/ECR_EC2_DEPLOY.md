# ECR push + EC2 hosting (recipe optimization agent)

How the Capstone **recipe opt agent** gets from git into a runnable staging site on AWS.

Supabase remains the database. This doc covers **container build/push** and **on-demand EC2 demos** only. For S3 data/artifacts and the older git+`uv` EC2 path, see [AWS_WORKFLOW.md](AWS_WORKFLOW.md).

## Big picture

```text
  developer push to `deployment`
            │
            ▼
  GitHub Actions (.github/workflows/push-ecr.yml)
            │  docker build (linux/amd64) → recipe_opt_web
            ▼
  Amazon ECR  repo: macroiq
            │  tags: :<git-sha>  and  :deployment
            │
            │  (manual — does NOT auto-start EC2)
            ▼
  EC2 staging (e.g. MacroIQDemo)
            │  docker pull + systemd (capstone-mvp-docker)
            ▼
  http://macroiq.org   (host :80 → container :8000 / recipe_opt_web)
```

| Piece | What it does | Cost when idle |
|--------|----------------|----------------|
| **GitHub Actions → ECR** | Builds the Docker image and stores it | ECR storage only (cents) |
| **EC2** | Runs that image for demos | **Stop the instance** → ~EBS only (~$3/mo for ~40 GB) |

Pushing to `deployment` updates the image in ECR. It does **not** start EC2 or change the live site until someone runs the deploy script (or starts the instance and pulls).

## Part 1 — Automated ECR push

### Trigger

- **Branch:** `deployment` (not `main` yet)
- **Events:** `push` to `deployment`, or manual **Run workflow**
- **Workflow file:** `.github/workflows/push-ecr.yml`

### What the workflow does

1. Checks out the repo  
2. Authenticates to AWS with GitHub Secrets `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`  
3. Logs into ECR in `us-east-1`  
4. Ensures ECR repository `macroiq` exists  
5. Builds `Dockerfile` for `linux/amd64`  
6. Pushes:
   - `956314528899.dkr.ecr.us-east-1.amazonaws.com/macroiq:<12-char-sha>`
   - `…/macroiq:deployment` (moving “current staging” tag)

### GitHub setup (once)

Repo → **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | IAM user that can push to ECR |
| `AWS_SECRET_ACCESS_KEY` | Matching secret |

IAM for that user needs ECR push permissions (create repo optional; push to `macroiq` required).

### Local / manual push (optional)

```bash
./scripts/deploy/push_mvp_ecr.sh
# or: ECR_REPO=macroiq AWS_REGION=us-east-1 ./scripts/deploy/push_mvp_ecr.sh
```

Same idea as CI; CI is the normal path on `deployment`.

### Image contents

- FastAPI app: `uvicorn recipe_opt_web.server:app` on port **8000** (recipe optimization agent UI)
- **Default page `/`:** MacroIQ product UI (`recipe_opt_web/static/macroiq.html` + `macroiq.css` / `macroiq.js`)
- **Playground `/playground`:** developer flow-graph UI (`static/index.html`)
- Health: `GET /health`
- Runtime deps only (`uv sync --frozen --no-dev`) — no notebook/pipeline/mvp extras
- **Caches in the image:**
  - FoodOn: `foodon_web/cache/{foodon_index,foodon_hierarchy,fdc_foodon_map}.json`
  - Dequant FDC grounding: `data/dequant_norm_llm_cache.json` (`DEQUANT_CACHE_PATH`)
- **Caches in Supabase (not baked into the image):**
  - Neighborhood Jaccard: `recipe.canonical_neighborhood_cache` — must match agent `NEIGHBORHOOD_CACHE_VERSION` (currently **2**). Populate with `scripts/populate_neighborhood_cache.py` / `precompute_canonical_neighborhoods.py` if lookups miss and rebuild live.
- Secrets are **not** baked in — supplied at runtime via `/opt/capstone/.env` on EC2
- Default `RECIPE_DATA_SOURCE=db` (Supabase) inside the container

## Part 2 — EC2 hosting (run the ECR image)

### Preferred path (Docker from ECR)

Scripts:

| Script | Role |
|--------|------|
| `scripts/deploy/deploy_ecr_to_ec2.sh` | From your laptop: start EC2, SCP unit, pull image, restart service |
| `scripts/deploy/remote_setup_ecr.sh` | Runs **on** the instance (install Docker if needed, ECR login, pull, systemd) |
| `infra/aws/capstone-mvp-docker.service` | systemd unit: `docker pull` + `docker run` on `:8000` |
| `scripts/deploy/stop_staging.sh` | Stop EC2 (save compute) |
| `scripts/deploy/status.sh` | Instance state + health hint |

Or via the combined staging entrypoint:

```bash
./scripts/deploy/deploy_staging.sh --start-ec2 --ecr
./scripts/deploy/deploy_staging.sh --start-ec2 --ecr --stop-after
```

### Local config (`deploy/aws.env`)

Copy from example (file is **gitignored**):

```bash
cp deploy/aws.env.example deploy/aws.env
```

Important fields:

```env
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=956314528899
S3_BUCKET_RAW=capstone-956314528899-raw
S3_BUCKET_ARTIFACTS=capstone-956314528899-artifacts

EC2_ENABLED=true
EC2_INSTANCE_ID=i-...
EC2_KEY_PATH=C:/Users/you/.../capstone-staging.pem
EC2_SSH_USER=ec2-user

ECR_REPO=macroiq
ECR_IMAGE_TAG=deployment
```

Never commit `deploy/aws.env`, `.env`, or `*.pem`.

### EC2 instance requirements (checklist)

| Item | Notes |
|------|--------|
| AMI | Amazon Linux 2023 |
| Size | `t3.small` (or `t3.medium` if memory-tight) |
| Disk | **~40 GB gp3** (8 GB is too small for Docker + image) |
| Key pair | `.pem` you can SSH with; restrict file ACLs on Windows |
| IAM role | e.g. `capstone-ec2-staging` with **`AmazonEC2ContainerRegistryPullOnly`** (or project policy via `attach_ec2_ecr_iam.sh`) |
| Security group | **SSH 22** + **TCP 80** (your IP and/or teammates’ IPs) |
| Public IP | Elastic IP recommended; then DNS `@` → EIP for `http://macroiq.org` |

### One-time: secrets on the instance

From the repo root (PowerShell or Git Bash), replace the IP:

```bash
ssh -i /path/to/capstone-staging.pem ec2-user@<PUBLIC_IP> \
  "sudo mkdir -p /opt/capstone && sudo chown ec2-user:ec2-user /opt/capstone"

scp -i /path/to/capstone-staging.pem .env ec2-user@<PUBLIC_IP>:/opt/capstone/.env
```

`.env` must include working Supabase `PG_*` pooler settings (see `.env.example`). Prefer transaction pooler: `PG_PSQL_USE_TRANSACTION_POOLER_PORT=1`.

### Day-to-day: on / off

**On (demo / test)** — Git Bash or WSL (scripts are bash):

```bash
cd /path/to/Capstone
./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2
# confirm Y — starts instance if stopped, pulls macroiq:deployment, restarts container
```

Open: `http://<public-ip>:8000`  
(Get IP from AWS console or `./scripts/deploy/status.sh`.)

**Off (save money):**

```bash
./scripts/deploy/stop_staging.sh
# or Stop instance in the EC2 console
```

### Cost ballpark (us-east-1)

| State | Rough cost |
|--------|------------|
| EC2 **running** 24/7 (`t3.small` + public IPv4 + ~40 GB disk) | ~**$20–22/mo** |
| EC2 **stopped** (disk only) | ~**$3/mo** |
| ECR image storage (~0.6 GB) | cents/mo |
| GitHub Actions builds | GitHub minutes only |

## Custom URL — `http://macroiq.org` (no `:8000`)

Browsers use **port 80** by default. The container still listens on **8000 inside Docker**; the host maps **`80:8000`**.

1. **Cloudflare DNS** — A record Name `@` → Elastic IP, **DNS only** (grey cloud).  
2. **Security group** — inbound **TCP 80** (your/partners’ IPs). Port 8000 is optional now.  
3. **systemd** — `infra/aws/capstone-mvp-docker.service` publishes `-p 80:8000`. Redeploy so the instance picks up the unit:
   ```bash
   ./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2
   ```
4. Open **http://macroiq.org** (and `/health`).

HTTPS (`https://macroiq.org`) needs a cert later (Let’s Encrypt or Cloudflare proxy) — not required for demos.

You already own **`macroiq.org`**. Prefer apex DNS + port **80** mapping (section above). Keep the options table for teammates deciding HTTPS later.

### Domain cost notes

| Item | Typical cost |
|------|----------------|
| Domain (e.g. `macroiq.org` — already purchased) | ~**$10–15 / year** |
| Elastic IP | ~**$0** while on a **running** instance; may charge if allocated while instance stopped |
| HTTPS via **ALB + ACM** | ACM free; **ALB** often ~**$16+/mo** — poor fit for stop/start demos |
| HTTPS via **Let’s Encrypt** on EC2 | Free; renewals need instance up occasionally |

### Options overview

| Option | Example URL | Notes |
|--------|-------------|--------|
| **A. Apex + port 80** (current goal) | `http://macroiq.org` | A `@` → EIP; Docker `-p 80:8000`; SG TCP 80 |
| **B. Explicit :8000** | `http://macroiq.org:8000` | Only if you keep `-p 8000:8000` |
| **C. sslip.io** | `http://EIP.sslip.io` | Free fallback without domain |
| **D. ALB + ACM HTTPS** | `https://macroiq.org` | Ongoing ALB cost — skip for Capstone demos |

```env
STAGING_PUBLIC_URL=http://macroiq.org
```

Share that URL with partners; open **TCP 80** to their IPs in the security group.

## Legacy path (not ECR)

`load_to_ebs.sh` / `remote_setup.sh` still support **git pull + `uv` + systemd** without Docker. Prefer **`--ecr`** / `deploy_ecr_to_ec2.sh` so demos match the GitHub-built image.

## Related files

| Path | Purpose |
|------|---------|
| `Dockerfile` | Agent image (`recipe_opt_web` + FoodOn caches) |
| `.github/workflows/push-ecr.yml` | CI build → ECR on `deployment` pushes |
| `.github/workflows/deploy-runtime.yml` | Manual checklist only (no auto live deploy) |
| `recipe_opt_web/server.py` | FastAPI app; `GET /health` |
| `infra/aws/capstone-mvp-docker.service` | EC2: pull ECR image, map host `:80` → `:8000` |

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Actions fails on AWS login | Check GitHub `AWS_*` secrets / IAM ECR push |
| Actions fails missing `.python-version` | Already fixed in Dockerfile (do not COPY missing file) |
| Deploy waits on `/health` forever | Add SG inbound **8000**; confirm `.env` on instance; `sudo systemctl status capstone-mvp-docker` |
| `password authentication failed` (Supabase) | Fix `PG_PASSWORD` / use pool user `postgres.<ref>` + port **6543** |
| SSH “UNPROTECTED PRIVATE KEY” (Windows) | Restrict `.pem` ACL to your user only (`icacls … /inheritance:r` then grant read) |
| Site OK yesterday, URL dead today | Instance stopped, or **public IP changed** after start — check console |

## Quick reference

```text
Code change → push `deployment` → Actions builds → ECR :deployment updated
                                                      ↓
                              (when you want a demo)
                         deploy_ecr_to_ec2.sh --start-ec2
                                                      ↓
                                         http://<ip>:8000
                                                      ↓
                                      stop_staging.sh when done
```
