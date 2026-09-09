#!/usr/bin/env python3
"""Generate notebooks/colab_ingredient_resolution_oss.ipynb (run once locally)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab_ingredient_resolution_oss.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


OSS_PROBE_AND_MODEL = r'''
import json
import os
import re
import threading
import asyncio
from dataclasses import dataclass, asdict
from typing import Any

import psutil
import torch

PROBE: dict[str, Any] = {}


def _gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def probe_memory() -> dict[str, Any]:
    """Set judge concurrency and embedding batch from GPU VRAM + system RAM."""
    vram = _gpu_vram_gb()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    cfg: dict[str, Any] = {
        "gpu_name": gpu_name,
        "vram_gb": round(vram, 1),
        "ram_gb": round(ram_gb, 1),
        "backend": "transformers",
        "judge_concurrency": 2,
        "vllm_max_num_seqs": 0,
        "max_new_tokens_judge": 1024,
        "max_new_tokens_enrichment": 1024,
        "max_new_tokens_portion": 512,
        "max_new_tokens_cap": 2048,
        "embed_batch": 0,
    }

    # Prefer vLLM on A100-class GPUs
    if vram >= 38:
        cfg["backend"] = "vllm"
        cfg["vllm_max_num_seqs"] = 8
        cfg["judge_concurrency"] = 8
        if vram >= 70:
            cfg["vllm_max_num_seqs"] = 12
            cfg["judge_concurrency"] = 12
    elif vram >= 14:
        cfg["backend"] = "transformers"
        cfg["judge_concurrency"] = 4
    else:
        cfg["backend"] = "transformers"
        cfg["judge_concurrency"] = 2

    if ram_gb >= 50:
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        cfg["embed_batch"] = min(512, max(128, int(free_gb * 48)))
    elif ram_gb >= 25:
        cfg["embed_batch"] = 256
    else:
        cfg["embed_batch"] = 0  # use pre-built S3 embeddings only

    min_parallel = int(os.environ.get("COLAB_MIN_BATCH", "8"))
    if cfg["backend"] == "vllm":
        cfg["vllm_max_num_seqs"] = max(min_parallel, cfg["vllm_max_num_seqs"])
        cfg["judge_concurrency"] = max(min_parallel, cfg["judge_concurrency"])
    elif vram >= 14:
        cfg["judge_concurrency"] = max(min_parallel, cfg["judge_concurrency"])
    elif min_parallel > cfg["judge_concurrency"]:
        print(
            f"WARNING: GPU VRAM {vram:.1f}GB — judge concurrency stays at "
            f"{cfg['judge_concurrency']} (set COLAB_MIN_BATCH lower or use A100 + vLLM for 8+)",
            flush=True,
        )

    return cfg


PROBE = probe_memory()
print(json.dumps(PROBE, indent=2))


def openai_schema_inner(response_schema: dict[str, Any]) -> dict[str, Any]:
    """Inner JSON Schema from OpenAI response_format wrapper."""
    if "schema" in response_schema:
        return response_schema["schema"]
    return response_schema


def parse_json_output(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(m.group(0))


@dataclass
class GenResult:
    parsed: dict[str, Any]
    raw: str
    prompt_tokens: int
    completion_tokens: int


class OssJsonLlm:
    """Open-source JSON LLM for judge / enrichment / portion pick."""

    def __init__(self, model_id: str, probe: dict[str, Any]):
        self.model_id = model_id
        self.probe = probe
        self.backend = probe.get("backend", "transformers")
        self._lock = threading.Lock()
        self._llm = None
        self._model = None
        self._tokenizer = None
        self._load()

    def _vllm_model_id(self) -> str:
        """Prefer AWQ weights for vLLM (fits A100 40GB with batching)."""
        if self.model_id.endswith("-AWQ"):
            return self.model_id
        if "14B" in self.model_id:
            return "Qwen/Qwen2.5-14B-Instruct-AWQ"
        if "7B" in self.model_id:
            return "Qwen/Qwen2.5-7B-Instruct-AWQ"
        return self.model_id

    def _load(self) -> None:
        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams  # noqa: F401

                max_seqs = int(self.probe.get("vllm_max_num_seqs", 8))
                vllm_id = self._vllm_model_id()
                self._llm = LLM(
                    model=vllm_id,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.90,
                    max_model_len=8192,
                    max_num_seqs=max_seqs,
                    dtype="auto",
                )
                self._vllm_model_id_loaded = vllm_id
                self._SamplingParams = SamplingParams
                print(f"Loaded vLLM {vllm_id} max_num_seqs={max_seqs}")
                return
            except Exception as exc:
                print(f"vLLM load failed ({exc}); falling back to transformers")
                self.backend = "transformers"

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"Loaded transformers 4-bit {self.model_id}")

    def _chat_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.backend == "vllm":
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        return len(tok.encode(text))

    def _generate_raw(
        self,
        prompt: str,
        max_new_tokens: int,
        inner_schema: dict[str, Any] | None,
    ) -> tuple[str, str | None]:
        """Single generation; vLLM uses StructuredOutputsParams(json=...) when schema set."""
        if self.backend == "vllm":
            sp_kw: dict[str, Any] = {"temperature": 0.0, "max_tokens": max_new_tokens}
            if inner_schema is not None:
                from vllm.sampling_params import StructuredOutputsParams

                sp_kw["structured_outputs"] = StructuredOutputsParams(json=inner_schema)
            params = self._SamplingParams(**sp_kw)
            with self._lock:
                outs = self._llm.generate([prompt], params, use_tqdm=False)
            out = outs[0].outputs[0]
            finish_reason = getattr(out, "finish_reason", None)
            return out.text.strip(), finish_reason

        def _gen_transformers() -> tuple[str, str | None]:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            gen_kw: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if inner_schema is not None:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )

                parser = JsonSchemaParser(inner_schema)
                gen_kw["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(
                    parser, self._tokenizer
                )
            with torch.inference_mode():
                out = self._model.generate(**inputs, **gen_kw)
            new_tokens = out[0, inputs["input_ids"].shape[1] :]
            raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            finish_reason = "length" if int(new_tokens.shape[-1]) >= max_new_tokens else "stop"
            return raw, finish_reason

        with self._lock:
            return _gen_transformers()

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        max_new_tokens: int = 512,
        extra_user: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenResult:
        inner_schema = openai_schema_inner(json_schema) if json_schema else None
        token_cap = int(self.probe.get("max_new_tokens_cap", 2048))
        tokens = max_new_tokens
        last_exc: Exception | None = None

        for attempt in range(3):
            full_user = user + (f"\n\n{extra_user}" if extra_user else "")
            if inner_schema is None:
                full_user += "\n\nRespond with a single JSON object only. No markdown fences."
            prompt = self._chat_prompt(system, full_user)
            prompt_tokens = self._count_tokens(prompt)

            try:
                raw, finish_reason = self._generate_raw(prompt, tokens, inner_schema)
                completion_tokens = self._count_tokens(raw)
                parsed = parse_json_output(raw)
                truncated = finish_reason == "length" or completion_tokens >= tokens - 1
                if truncated and tokens < token_cap and attempt < 2:
                    tokens = min(tokens * 2, token_cap)
                    continue
                if truncated:
                    raise ValueError(
                        f"model output truncated at max_tokens={tokens} "
                        f"(finish_reason={finish_reason!r})"
                    )
                return GenResult(parsed, raw, prompt_tokens, completion_tokens)
            except Exception as exc:
                last_exc = exc
                if attempt < 2 and tokens < token_cap:
                    tokens = min(tokens * 2, token_cap)
                    continue
                raise

        raise last_exc or RuntimeError("generate_json failed")

    async def generate_json_async(self, *args, **kwargs) -> GenResult:
        return await asyncio.to_thread(self.generate_json, *args, **kwargs)

    def meta(self) -> dict[str, Any]:
        meta = {
            "model_id": self.model_id,
            "backend": self.backend,
            "probe": self.probe,
        }
        if hasattr(self, "_vllm_model_id_loaded"):
            meta["vllm_model_id"] = self._vllm_model_id_loaded
        return meta


DEFAULT_MODEL = os.environ.get("OSS_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct")
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"

try:
    OSS_LLM = OssJsonLlm(DEFAULT_MODEL, PROBE)
except Exception as exc:
    print(f"OOM loading {DEFAULT_MODEL}: {exc}; trying {FALLBACK_MODEL}")
    min_parallel = int(os.environ.get("COLAB_MIN_BATCH", "8"))
    if PROBE.get("vram_gb", 0) >= 38:
        PROBE["judge_concurrency"] = max(min_parallel, PROBE.get("judge_concurrency", min_parallel))
        if PROBE.get("backend") == "vllm":
            PROBE["vllm_max_num_seqs"] = max(min_parallel, PROBE.get("vllm_max_num_seqs", min_parallel))
    else:
        PROBE["judge_concurrency"] = min(PROBE.get("judge_concurrency", 4), 4)
        print(
            "WARNING: 7B fallback on small GPU — concurrency may be below COLAB_MIN_BATCH",
            flush=True,
        )
    OSS_LLM = OssJsonLlm(FALLBACK_MODEL, PROBE)

# Warmup: structured JSON call (validates vLLM StructuredOutputsParams / lm-format-enforcer)
_warmup_schema = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_r = OSS_LLM.generate_json(
    "You output JSON only.",
    "Return ok=true.",
    max_new_tokens=64,
    json_schema=_warmup_schema,
)
print("OSS backend ready:", OSS_LLM.backend, OSS_LLM.model_id, "warmup=", _r.parsed)
'''.strip() + "\n"


PATCH_AND_RUN = r'''
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import ingredient_match_llm as iml
import ingredient_match_llm_portion as iml_portion
import line_enrichment_llm as lel
import portion_resolve_llm as prl
import openai_fallback
from portion_pipeline_feasibility import run_feasibility

CACHE = Path("/content/capstone_cache")
OUT = Path("/content/capstone_runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT.mkdir(parents=True, exist_ok=True)
RUN_ID = OUT.name

os.environ["FOOD_4MACRO_CACHE"] = str(CACHE / "food_4macro.csv")
os.environ["CAPSTONE_RECIPE_CACHE"] = str(CACHE / "recipe_cache")

# Dummy OpenAI clients (unused after patches)
class _Dummy:
    class chat:
        class completions:
            @staticmethod
            async def create(**kwargs):
                raise RuntimeError("OpenAI client should not be called")


openai_fallback.get_async_openai_client = lambda: _Dummy()
openai_fallback.get_sync_openai_client = lambda: _Dummy()


async def oss_judge_async(
    client,
    model: str,
    user_prompt: str,
    valid_fdc_ids: set[int],
    *,
    system_prompt: str | None = None,
):
    sys_prompt = system_prompt or iml_portion.SYSTEM_PROMPT
    prompt_tokens = completion_tokens = 0
    error = None
    parsed: dict = {}
    raw_response = None

    try:
        r = await OSS_LLM.generate_json_async(
            sys_prompt,
            user_prompt,
            max_new_tokens=PROBE.get("max_new_tokens_judge", 1024),
            json_schema=iml.RESPONSE_SCHEMA,
        )
        parsed, raw_response = r.parsed, r.raw
        prompt_tokens += r.prompt_tokens
        completion_tokens += r.completion_tokens
        fdc_id = parsed.get("fdc_id")
        if fdc_id is not None and int(fdc_id) not in valid_fdc_ids:
            hint = (
                "Your previous fdc_id was not in the candidate list. "
                "Choose only from the listed fdc_id values, or null."
            )
            r2 = await OSS_LLM.generate_json_async(
                sys_prompt,
                user_prompt,
                max_new_tokens=PROBE.get("max_new_tokens_judge", 1024),
                json_schema=iml.RESPONSE_SCHEMA,
                extra_user=hint,
            )
            parsed, raw_response = r2.parsed, r2.raw
            prompt_tokens += r2.prompt_tokens
            completion_tokens += r2.completion_tokens
            fdc_id = parsed.get("fdc_id")
            if fdc_id is not None and int(fdc_id) not in valid_fdc_ids:
                error = "invalid_fdc_id"
                parsed["fdc_id"] = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    matched_portion_id = parsed.get("matched_portion_id")
    if matched_portion_id is not None:
        try:
            matched_portion_id = int(matched_portion_id)
        except (TypeError, ValueError):
            matched_portion_id = None

    return {
        "fdc_id": parsed.get("fdc_id"),
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "matched_portion_id": matched_portion_id,
        "negligible_calories": bool(parsed.get("negligible_calories", False)),
        "response": raw_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }


async def oss_enrich_one_async(client, model: str, ingredient: str, rules_plan):
    from line_enrichment_llm import build_user_prompt, validate_response

    prompt_tokens = completion_tokens = 0
    error = None
    parsed: dict = {}
    raw_response = None

    try:
        user = build_user_prompt(ingredient, rules_plan)
        r = await OSS_LLM.generate_json_async(
            lel.SYSTEM_PROMPT,
            user,
            max_new_tokens=PROBE.get("max_new_tokens_enrichment", 1024),
            json_schema=lel.RESPONSE_SCHEMA,
        )
        parsed, raw_response = r.parsed, r.raw
        prompt_tokens += r.prompt_tokens
        completion_tokens += r.completion_tokens
        validation_error = validate_response(parsed)
        if validation_error:
            r2 = await OSS_LLM.generate_json_async(
                lel.SYSTEM_PROMPT,
                user,
                max_new_tokens=PROBE.get("max_new_tokens_enrichment", 1024),
                json_schema=lel.RESPONSE_SCHEMA,
                extra_user=f"Validation failed ({validation_error}). Fix and resubmit.",
            )
            parsed, raw_response = r2.parsed, r2.raw
            prompt_tokens += r2.prompt_tokens
            completion_tokens += r2.completion_tokens
            if validate_response(parsed):
                error = validation_error
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    from ingredient_parse_llm import normalize_ingredient_key

    return {
        "ingredient_norm": normalize_ingredient_key(ingredient),
        "ingredient_raw": ingredient,
        "enrichment": parsed if not error else {},
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "error": error,
        "response": raw_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "price_estimate_usd": 0.0,
    }


def oss_pick_portion_sync(
    model: str,
    *,
    ingredient: str,
    quantity: float,
    unit: str | None,
    name: str | None,
    amount_kind: str,
    fdc_id: int,
    raw_rows: list,
):
    block = prl.format_portion_options(raw_rows)
    user = prl.build_user_prompt(
        ingredient=ingredient,
        quantity=quantity,
        unit=unit,
        name=name,
        amount_kind=amount_kind,
        fdc_id=fdc_id,
        portion_block=block,
    )
    r = OSS_LLM.generate_json(
        prl.SYSTEM_PROMPT,
        user,
        max_new_tokens=PROBE.get("max_new_tokens_portion", 512),
        json_schema=prl.RESPONSE_SCHEMA,
    )
    parsed = r.parsed
    return {
        "portion_id": parsed.get("portion_id"),
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "price_estimate_usd": 0.0,
        "user_prompt": user,
    }


iml.judge_async = oss_judge_async
lel.enrich_one_async = oss_enrich_one_async
prl.pick_portion_sync = oss_pick_portion_sync

limit = int(os.environ["COLAB_LIMIT"]) if os.environ.get("COLAB_LIMIT") else None
model_label = OSS_LLM.model_id
min_parallel = int(os.environ.get("COLAB_MIN_BATCH", "8"))
judge_concurrency = max(min_parallel, int(PROBE.get("judge_concurrency", min_parallel)))

bucket = os.environ.get("S3_BUCKET_ARTIFACTS", "")
if bucket:
    os.environ["COLAB_PROGRESS_S3_PREFIX"] = f"s3://{bucket}/colab/runs/{RUN_ID}/"

recipe_csv = CACHE / "RecipeNLG.csv"

import nest_asyncio
nest_asyncio.apply()

report = run_feasibility(
    n_recipes=1000,
    seed=42,
    model=model_label,
    out_dir=OUT,
    baseline_dir=None,
    food_cache_dir=CACHE / "food_cache",
    limit=limit,
    concurrency=judge_concurrency,
    skip_portion_llm=False,
    force_amount=False,
    force_judging=False,
    force_payloads=False,
    force_all=False,
    finalize_only=False,
    only_no_portion=False,
    use_mlflow=False,
    progress_mode="colab",
    disk_flush_every=10,
    judge_log_every=10_000,
    parquet_compact_every=10,
    enrichment_concurrency=judge_concurrency,
    sample_manifest=CACHE / "sampled_recipe_ids.json",
    recipe_csv=recipe_csv,
    recipe_cache_dir=CACHE / "recipe_cache",
)

(OUT / "oss_model_meta.json").write_text(
    json.dumps({**OSS_LLM.meta(), "run_id": RUN_ID, "colab_limit": limit}, indent=2) + "\n"
)
print("Run complete:", OUT)
print(json.dumps({k: report[k] for k in (
    "n_lines", "fdc_match_rate_all", "gram_resolve_rate_all",
    "fdc_and_gram_rate_all", "judge_error_count", "elapsed_sec",
) if k in report}, indent=2))
'''.strip() + "\n"


cells = [
    md(
        "# Colab OSS ingredient resolution pipeline\n\n"
        "Runs the full `portion_pipeline_feasibility` pipeline on the canonical **1,000 recipes** "
        "(seed 42) using an open-source Hugging Face model instead of OpenAI.\n\n"
        "**Prerequisites:** Colab GPU runtime (A100 recommended), Supabase credentials, AWS S3 access, "
        "and the input bundle at `s3://{artifacts}/colab/feasibility_1000_seed42/`.\n\n"
        "See [`docs/COLAB_OSS_RESOLUTION.md`](../docs/COLAB_OSS_RESOLUTION.md) for setup."
    ),
    md(
        "## Progress\n\n"
        "The pipeline cell shows **one tqdm bar per LLM pass** (enrichment, judging, portion). "
        "`progress.json` under `OUT` updates every 10 prompts with live stats "
        "(and syncs to S3 when `COLAB_PROGRESS_S3_PREFIX` is set)."
    ),
    code(
        "import os\n"
        "os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'\n"
        "os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'\n"
        "os.environ['TRANSFORMERS_VERBOSITY'] = 'error'"
    ),
    code("!nvidia-smi\n!free -h"),
    code(
        "%%capture\n"
        "!pip install -q transformers accelerate bitsandbytes sentence-transformers \\\n"
        "  pandas pyarrow psycopg2-binary rapidfuzz ingredient-parser-nlp \\\n"
        "  faiss-cpu scikit-learn tqdm psutil networkx python-dotenv boto3 nest_asyncio \\\n"
        "  lm-format-enforcer"
    ),
    code(
        "%%capture\n"
        "!pip uninstall -y vllm\n"
        "!pip install -q vllm==0.23.0 \\\n"
        "  --extra-index-url https://wheels.vllm.ai/0.23.0/cu129 \\\n"
        "  --extra-index-url https://download.pytorch.org/whl/cu128"
    ),
    code(
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "import torch\n"
        "from vllm import LLM\n"
        "print('OK', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    ),
    code(
        "import os\n"
        "from google.colab import userdata\n\n"
        "def _secret(key: str, default: str = '') -> str:\n"
        "    try:\n"
        "        return userdata.get(key)\n"
        "    except Exception:\n"
        "        return os.environ.get(key, default)\n\n"
        "for key in (\n"
        "    'PG_POOL_USER', 'PG_PASSWORD', 'PG_POOL_HOST', 'PG_POOL_SESSION_PORT',\n"
        "    'PG_DATABASE', 'PG_SSL_MODE', 'HF_TOKEN',\n"
        "    'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION',\n"
        "    'S3_BUCKET_ARTIFACTS', 'S3_BUCKET_RAW',\n"
        "):\n"
        "    val = _secret(key)\n"
        "    if val:\n"
        "        os.environ[key] = val\n\n"
        "os.environ.setdefault('PG_POOL_SESSION_PORT', '5432')\n"
        "os.environ.setdefault('PG_DATABASE', 'postgres')\n"
        "os.environ.setdefault('PG_SSL_MODE', 'require')\n"
        "os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')\n"
        "os.environ.setdefault('OSS_MODEL_ID', 'Qwen/Qwen2.5-14B-Instruct')\n"
        "print('Secrets loaded. S3 bucket:', os.environ.get('S3_BUCKET_ARTIFACTS'))"
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n\n"
        "REPO = Path('/content/Capstone')\n"
        "if REPO.is_dir():\n"
        "    !cd /content/Capstone && git fetch origin && git checkout agent_mvp && git pull origin agent_mvp\n"
        "else:\n"
        "    !git clone -b agent_mvp https://github.com/dcosta224/Capstone.git /content/Capstone\n"
        "%cd /content/Capstone\n"
        "sys.path.insert(0, str(REPO / 'scripts'))"
    ),
    code(
        "import os\n"
        "from pathlib import Path\n\n"
        "from colab_s3 import download_s3_file, sync_s3_prefix\n\n"
        "CACHE = Path('/content/capstone_cache')\n"
        "CACHE.mkdir(parents=True, exist_ok=True)\n"
        "bucket = os.environ['S3_BUCKET_ARTIFACTS']\n"
        "raw_bucket = os.environ.get('S3_BUCKET_RAW', '')\n\n"
        "sync_s3_prefix(bucket, 'colab/feasibility_1000_seed42/', CACHE)\n\n"
        "recipe_csv = CACHE / 'RecipeNLG.csv'\n"
        "if not recipe_csv.is_file():\n"
        "    if not raw_bucket:\n"
        "        raise SystemExit('Set S3_BUCKET_RAW or place RecipeNLG.csv in the cache')\n"
        "    download_s3_file(raw_bucket, 'Data/recipes/RecipeNLG.csv', recipe_csv)\n"
        "print('Cache ready:', len(list(CACHE.iterdir())), 'items')"
    ),
    code(
        "import os\n"
        "from pathlib import Path\n\n"
        "src = Path('/content/capstone_cache/RecipeNLG.csv')\n"
        "dst_dir = Path('/content/Capstone/Data/recipes')\n"
        "dst_dir.mkdir(parents=True, exist_ok=True)\n"
        "dst = dst_dir / 'RecipeNLG.csv'\n"
        "if not src.is_file():\n"
        "    raise SystemExit('RecipeNLG.csv missing from cache')\n"
        "if dst.exists() or dst.is_symlink():\n"
        "    dst.unlink()\n"
        "os.symlink(src, dst)\n"
        "print('Linked', dst, '->', src)"
    ),
    code(OSS_PROBE_AND_MODEL),
    code(PATCH_AND_RUN),
    code(
        "from pathlib import Path\n\n"
        "from colab_s3 import upload_dir_to_s3\n\n"
        "bucket = os.environ['S3_BUCKET_ARTIFACTS']\n"
        "prefix = f'colab/runs/{RUN_ID}/'\n"
        "upload_dir_to_s3(OUT, bucket, prefix)\n"
        "print('Uploaded to', f's3://{bucket}/{prefix}')"
    ),
    code(
        "import json\n"
        "from pathlib import Path\n\n"
        "baseline = json.loads((CACHE / 'baseline_summary.json').read_text())\n"
        "new_report = json.loads((OUT / 'feasibility_report.json').read_text())\n\n"
        "keys = [\n"
        "    'fdc_match_rate_all', 'gram_resolve_rate_all', 'fdc_and_gram_rate_all',\n"
        "    'fdc_and_gram_rate_needs_portion', 'judge_error_count', 'n_lines',\n"
        "]\n"
        "print('metric | baseline (GPT) | OSS')\n"
        "for k in keys:\n"
        "    b = baseline.get(k)\n"
        "    n = new_report.get(k)\n"
        "    print(f'{k}: {b} -> {n}')"
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(f"Wrote {OUT}")
