"""
Simple FastAPI-based miner server for Babelbit subnet.
Serves predictions via HTTP endpoint that validators can call directly.

This version uses an OpenAI-compatible HTTP API (e.g. official OpenAI,
vLLM / llama.cpp OpenAI server, etc.) instead of loading a local HF model.

Note: Run register_axon.py first to register your miner on-chain,
then run this script to serve predictions.
"""
import asyncio
import logging
import os
from traceback import format_exc
from typing import Optional, List

import aiohttp
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from babelbit.miner.utils import verify_bittensor_request
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from babelbit.utils.settings import get_settings

logger = logging.getLogger(__name__)

# Cached miner hotkey (SS58) for Bittensor request verification.
_MINER_HOTKEY_SS58: Optional[str] = None


def _get_miner_hotkey_ss58() -> Optional[str]:
    """
    Load and cache this miner's hotkey SS58 address.

    This is used to ensure that incoming Bittensor protocol requests are
    actually addressed to this miner (i.e., bt_header_axon_hotkey matches).
    """
    global _MINER_HOTKEY_SS58
    if _MINER_HOTKEY_SS58:
        return _MINER_HOTKEY_SS58
    try:
        settings = get_settings()
        keypair = load_hotkey_keypair(
            settings.BITTENSOR_WALLET_COLD,
            settings.BITTENSOR_WALLET_HOT,
        )
        _MINER_HOTKEY_SS58 = keypair.ss58_address
        return _MINER_HOTKEY_SS58
    except Exception as e:
        logger.warning(f"Unable to load miner hotkey for request verification: {e}")
        return None


# ---------------------------------------------------------------------------
# Client IP helpers
# ---------------------------------------------------------------------------

def get_client_ip(request: Request) -> str:
    """
    Resolve the best-guess client IP address from an incoming FastAPI Request.

    This is *proxy-aware*, i.e. it prefers common forwarding headers when
    the app is running behind a reverse proxy or load balancer.

    Priority:
      1. X-Forwarded-For (first IP in the comma-separated list)
      2. X-Real-IP
      3. request.client.host (direct connection from ASGI server)

    Returns:
        A string containing the resolved IP address, or "unknown" if it
        cannot be determined.
    """
    # 1) Standard header carrying the entire proxy chain:
    #    "client, proxy1, proxy2, ..."
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # Take the first entry (original client); strip whitespace to be safe.
        first = xff.split(",")[0].strip()
        if first:
            return first

    # 2) Common alternative header set by some proxies (e.g. NGINX).
    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip.strip()

    # 3) Fallback: direct peer IP as seen by the ASGI server (Uvicorn).
    if request.client:
        return request.client.host

    # If everything else fails, return a sentinel value instead of raising.
    return "unknown"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class BBUtteranceEvaluation(BaseModel):
    """
    Evaluation result for utterance prediction (echoed from validator if present).

    These values are not used by the miner itself but are kept for compatibility
    with the BBPredictedUtterance schema and may be populated by validators.
    """
    lexical_similarity: float = 0.0
    semantic_similarity: float = 0.0
    earliness: float = 0.0
    u_step: float = 0.0


class PredictRequest(BaseModel):
    """
    Request schema matching chute template / BBPredictedUtterance.

    The miner only *uses* index, step, prefix, context for prediction.
    Other fields are kept for compatibility (and may be filled by validators).
    """
    index: str  # UUID session identifier (originates from utterance engine)
    step: int   # Step index within the current utterance
    prefix: str
    context: str = ""
    done: bool = False
    ground_truth: str | None = None
    prediction: str = ""
    evaluation: BBUtteranceEvaluation | None = None


class PredictResponse(BaseModel):
    """
    Simple response schema expected by validator.

    The validator only needs the prediction string; all scoring and bookkeeping
    is handled on the validator side.
    """
    prediction: str


# ---------------------------------------------------------------------------
# OpenAI-compatible miner implementation
# ---------------------------------------------------------------------------

def _classify_model_type(model_name: str, explicit: Optional[str]) -> str:
    """
    Decide whether the model should be treated as "base" or "instruct".

    Priority:
      1. If explicit is "base" or "instruct", trust it (OPENAI_MODEL_TYPE).
      2. Otherwise, infer from model name heuristics.

    Heuristics:
      - If the model name contains "gpt-4", "gpt-3.5", "instruct", "chat",
        or "turbo", we treat it as an instruction/chat model and use the
        /chat/completions endpoint.
      - Otherwise, we fall back to "base" and use the /completions endpoint.
    """
    if explicit:
        value = explicit.lower()
        if value in ("base", "instruct"):
            logger.info("OPENAI_MODEL_TYPE explicitly set to '%s'", value)
            return value
        else:
            logger.warning(
                "Unknown OPENAI_MODEL_TYPE=%s, will auto-detect from model name",
                explicit,
            )

    name = (model_name or "").lower()

    # Heuristics for instruction/chat models
    instruct_markers = [
        "gpt-4",
        "gpt-3.5",
        "instruct",
        "chat",
        "turbo",
    ]

    if any(m in name for m in instruct_markers):
        logger.info(
            "Detected model '%s' as INSTRUCT model based on name heuristics",
            model_name,
        )
        return "instruct"

    # Fallback: treat as base model
    logger.info(
        "Detected model '%s' as BASE model (no instruct/chat markers found)",
        model_name,
    )
    return "base"


def _normalize_base_url(base_url: str) -> str:
    """
    Normalize OPENAI_BASE_URL / OPENAI_API_BASE for endpoint construction.

    Accepts either:
      - https://api.openai.com/v1
      - https://api.openai.com
      - https://<provider>/v1

    Returns:
        A base URL that *always* ends with '/v1', so we can safely append
        '/chat/completions' or '/completions'.
    """
    u = (base_url or "").strip().rstrip("/")
    if not u:
        u = "https://api.openai.com/v1"
    if u.endswith("/v1"):
        return u
    return u + "/v1"


class BabelbitMiner:
    """
    Miner that serves predictions by calling an OpenAI-compatible HTTP API
    (e.g., official OpenAI, vLLM / llama.cpp with OpenAI proxy, etc.).

    Model types and endpoints:

      - "instruct":
          * Uses /chat/completions with role-based messages.
          * Prompt structure matches original HuggingFace-based miner:
            system + user (Context + prefix).
      - "base":
          * Uses /completions with a single `prompt`:
            context sentences joined by '\\n', followed by prefix.

    In both cases, the dialogue context string is first split into
    sentences by delimiter " EOF ", then merged into a string list
    which is placed into the prompt (newline-joined).
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        model_type: str,
        request_timeout: float = 10.0,
    ):
        """
        Initialize the miner with OpenAI-compatible API configuration.

        Args:
            api_base:
                Base URL of the OpenAI-compatible API, e.g.
                "https://api.openai.com" or "http://localhost:8000".
            api_key:
                API key / bearer token to use in Authorization header.
            model:
                Model name for /chat/completions or /completions.
            model_type:
                Either "base" or "instruct".
            request_timeout:
                Per-request timeout for the API call (seconds).
        """
        self.api_base = _normalize_base_url(api_base)
        self.api_key = api_key
        self.model = model
        self.model_type = model_type  # already normalized by classifier
        self.request_timeout = float(request_timeout)

        # A shared aiohttp client session for all requests.
        # Using a single session gives better connection reuse and lower latency.
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        logger.info("Initialized BabelbitMiner with OpenAI-compatible endpoint:")
        logger.info("  base=%s", self.api_base)
        logger.info("  model=%s", self.model)
        logger.info("  model_type=%s", self.model_type)
        logger.info("  timeout=%.2fs", self.request_timeout)

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Lazily create a shared aiohttp.ClientSession.

        By keeping one session per process, we avoid the overhead of
        creating TCP connections for every /predict call.
        """
        if self._session and not self._session.closed:
            return self._session

        async with self._session_lock:
            if self._session and not self._session.closed:
                return self._session

            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info("Created new aiohttp.ClientSession for OpenAI-compatible client")
            return self._session

    async def load(self):
        """
        Placeholder for compatibility with the old HF-based miner.

        For an HTTP-based OpenAI-compatible API we don't need to preload anything,
        but we validate configuration here so misconfigurations are caught early.
        """
        if not self.api_key:
            raise RuntimeError(
                "OpenAI-compatible API key is missing. "
                "Set OPENAI_API_KEY."
            )
        if not self.model:
            raise RuntimeError(
                "OpenAI-compatible model is missing. "
                "Set OPENAI_MODEL."
            )
        if self.model_type not in ("base", "instruct"):
            raise RuntimeError(
                f"Invalid model_type '{self.model_type}', must be 'base' or 'instruct'."
            )
        logger.info(
            "BabelbitMiner.load(): configuration looks valid "
            "(no local model to load)"
        )

    @staticmethod
    def _get_env_int(name: str, default: int) -> int:
        """Get integer from environment variable with a safe fallback."""
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    @staticmethod
    def _get_env_float(name: str, default: float) -> float:
        """Get float from environment variable with a safe fallback."""
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    @staticmethod
    def _split_context(context: str) -> List[str]:
        """
        Split the context string into sentences using delimiter " EOF ",
        strip whitespace, and drop empty chunks.

        Example:
            context = "S1 EOF S2 EOF S3"
            -> ["S1", "S2", "S3"]

        Returns:
            A list of context sentences.
        """
        if not context:
            return []
        parts = context.split(" EOF ")
        sentences: List[str] = []
        for p in parts:
            s = p.strip()
            if s:
                sentences.append(s)
        return sentences

    @staticmethod
    def _get_stop_sequences() -> List[str]:
        """
        Build stop sequence list from OPENAI_STOP_SEQS environment variable
        (comma-separated). If not set, use a default that is suitable for
        utterance-level completion.

        Example:
            OPENAI_STOP_SEQS=" EOF,\\n\\n"

        Notes:
            The values are unescaped using 'unicode_escape' so you can pass
            sequences like "\\n\\n" via environment variables.
        """
        raw = os.getenv("OPENAI_STOP_SEQS", "").strip()
        if not raw:
            # Default: stop when the model tries to cross utterance boundary.
            return [" EOF"]

        items: List[str] = []
        for part in raw.split(","):
            s = part.encode("utf-8").decode("unicode_escape").strip()
            if s:
                items.append(s)
        return items or [" EOF"]

    async def _call_openai_chat(self, messages) -> str:
        """
        Call the OpenAI-compatible /chat/completions endpoint and return
        the raw model content (assistant message text).

        Args:
            messages:
                List of {"role": "...", "content": "..."} dicts to send as the
                chat history to the API.
        """
        session = await self._get_session()

        max_new_tokens = self._get_env_int("CHUTE_MAX_NEW_TOKENS", 24)
        temperature = self._get_env_float("CHUTE_TEMPERATURE", 0.7)
        top_p = self._get_env_float("CHUTE_TOP_P", 0.95)
        stop = self._get_stop_sequences()

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }

        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                text = await resp.text()

                if resp.status != 200:
                    logger.warning(
                        "OpenAI-compatible CHAT API returned non-200: %s, body=%s",
                        resp.status,
                        text[:300],
                    )
                    return ""

                try:
                    data = await resp.json()
                except Exception:
                    logger.error("Failed to parse JSON from OpenAI-compatible CHAT response")
                    return ""

                try:
                    choices = data.get("choices") or []
                    if not choices:
                        logger.warning("OpenAI-compatible CHAT response has no choices")
                        return ""
                    msg = choices[0].get("message") or {}
                    content = msg.get("content") or ""
                    return content
                except Exception as e:
                    logger.error(f"Error extracting content from CHAT response: {e}")
                    return ""
        except Exception as e:
            logger.error(f"Error calling OpenAI-compatible CHAT API: {e}")
            logger.debug("Traceback:\n%s", format_exc())
            return ""

    async def _call_openai_completion(self, prompt: str) -> str:
        """
        Call the OpenAI-compatible /completions endpoint (for base models)
        and return the raw text (continuation).

        Args:
            prompt:
                The full prompt string (context sentences joined by '\\n' +
                current prefix).
        """
        session = await self._get_session()

        max_new_tokens = self._get_env_int("CHUTE_MAX_NEW_TOKENS", 24)
        temperature = self._get_env_float("CHUTE_TEMPERATURE", 0.7)
        top_p = self._get_env_float("CHUTE_TOP_P", 0.95)
        stop = self._get_stop_sequences()

        url = f"{self.api_base}/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }

        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                text = await resp.text()

                if resp.status != 200:
                    logger.warning(
                        "OpenAI-compatible COMPLETIONS API returned non-200: %s, body=%s",
                        resp.status,
                        text[:300],
                    )
                    return ""

                try:
                    data = await resp.json()
                except Exception:
                    logger.error("Failed to parse JSON from COMPLETIONS response")
                    return ""

                try:
                    choices = data.get("choices") or []
                    if not choices:
                        logger.warning("COMPLETIONS response has no choices")
                        return ""
                    content = choices[0].get("text") or ""
                    return content
                except Exception as e:
                    logger.error(f"Error extracting content from COMPLETIONS response: {e}")
                    return ""
        except Exception as e:
            logger.error(f"Error calling OpenAI-compatible COMPLETIONS API: {e}")
            logger.debug("Traceback:\n%s", format_exc())
            return ""

    async def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Generate prediction for the given prefix and context, using an
        OpenAI-compatible API (chat or completions) based on model_type.

        Pipeline:
          1. Split `request.context` into sentences using " EOF ".
          2. For "base" models:
                prompt = "\\n".join(context_sentences + [prefix])
             and call /completions.
          3. For "instruct" models:
                messages = [system, user(context+prefix)]
             and call /chat/completions.
          4. Concatenate original prefix + generated completion as the final
             "full utterance" prediction.
        """
        try:
            # Ensure configuration is valid (no-op after first successful call).
            await self.load()

            if not request.prefix:
                logger.warning("No prefix provided, returning empty prediction")
                return PredictResponse(prediction="")

            # Split context into a list of sentences using " EOF ".
            context_sentences = self._split_context(request.context)

            logger.info("Generating prediction for prefix: %r", request.prefix)
            logger.info("Using %d context sentence(s)", len(context_sentences))

            if self.model_type == "base":
                # BASE MODEL FLOW -> /completions (prompt continuation)
                # Prompt is just the concatenation of all context sentences
                # and the current prefix, separated by newlines.
                if context_sentences:
                    prompt_text = "\n".join(context_sentences + [request.prefix])
                else:
                    prompt_text = request.prefix

                completion = await self._call_openai_completion(prompt=prompt_text)
            else:
                # INSTRUCT MODEL FLOW -> /chat/completions (system + user)
                system_msg = (
                    "You are a helpful assistant that completes the current utterance naturally "
                    "and succinctly. "
                    "Return only the completed utterance text without quotes or extra commentary."
                )

                if context_sentences:
                    context_block = "\n".join(context_sentences)
                    user_msg = (
                        "Context:\n"
                        f"{context_block}\n\n"
                        "Continue the utterance that begins with:\n"
                        f"{request.prefix}"
                    )
                else:
                    user_msg = (
                        "Continue the utterance that begins with:\n"
                        f"{request.prefix}"
                    )

                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]

                completion = await self._call_openai_chat(messages=messages)

            # If nothing useful, fallback.
            if not completion or not completion.strip():
                fallback = os.getenv("CHUTE_FALLBACK_COMPLETION", "...")
                full_prediction = f"{request.prefix} {fallback}".strip()
                logger.info("Empty completion, using fallback -> %r", full_prediction)
                return PredictResponse(prediction=full_prediction)

            # Clean and join prefix + completion.
            prediction = completion.strip()

            # Heuristic: if the model repeats the prefix at the beginning, strip it.
            if prediction.startswith(request.prefix):
                prediction = prediction[len(request.prefix):].lstrip()

            full_prediction = f"{request.prefix} {prediction}".strip()
            logger.info("Generated (truncated): %r", full_prediction[:100])
            return PredictResponse(prediction=full_prediction)

        except Exception as e:
            logger.error(f"Error in predict: {e}")
            logger.error(format_exc())
            # Return empty prediction on error (validator will handle).
            return PredictResponse(prediction="")


# Global miner instance (initialized on startup)
miner_instance: Optional[BabelbitMiner] = None


async def startup():
    """
    FastAPI startup event handler.

    Responsibilities:
      - Read OpenAI-compatible settings from environment.
      - Decide model_type ("base" vs "instruct") via heuristics or override.
      - Instantiate BabelbitMiner.
      - Run miner_instance.load() once to validate configuration early.
    """
    global miner_instance

    settings = get_settings()

    # Read OpenAI-compatible configuration only from OPENAI_* environment variables.
    api_base = (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com"
    )
    api_key = os.getenv("OPENAI_API_KEY") or ""
    model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    explicit_model_type = os.getenv("OPENAI_MODEL_TYPE")  # may be None
    request_timeout = float(os.getenv("OPENAI_TIMEOUT", "10.0"))

    model_type = _classify_model_type(model_name=model, explicit=explicit_model_type)

    logger.info("OpenAI-compatible miner configuration:")
    logger.info("  API base: %s", api_base)
    logger.info("  Model:    %s", model)
    logger.info("  Type:     %s", model_type)
    logger.info("  Timeout:  %.2fs", request_timeout)
    logger.info("  Stop seq: %s", BabelbitMiner._get_stop_sequences())
    logger.info("")

    if not api_key:
        logger.error(
            "OPENAI_API_KEY is not set. Miner cannot start without an API key."
        )
        raise RuntimeError("OpenAI-compatible API key is required")

    miner_instance = BabelbitMiner(
        api_base=api_base,
        api_key=api_key,
        model=model,
        model_type=model_type,
        request_timeout=request_timeout,
    )

    logger.info("Initializing OpenAI-compatible miner client...")
    try:
        await miner_instance.load()
        logger.info("✅ OpenAI-compatible miner ready (no local model loaded)")
        logger.info("")
    except Exception as e:
        logger.error("❌ Failed to initialize OpenAI-compatible miner!")
        logger.error("   Error: %s", e)
        logger.error("")
        raise


# Create FastAPI app and register startup hook.
app = FastAPI(title="Babelbit Miner", on_startup=[startup])


@app.get("/healthz")
async def health():
    """
    Health check endpoint (Kubernetes-style).

    Returns basic information about:
      - whether the miner instance is initialized
      - which model it is configured to call
      - which backend flavor is in use
    """
    return {
        "status": "healthy",
        "model": miner_instance.model if miner_instance else "not_initialized",
        "model_loaded": miner_instance is not None,
        "backend": "openai-compatible",
        "model_type": miner_instance.model_type if miner_instance else None,
    }


@app.get("/health")
async def health_alt():
    """
    Alternative health check endpoint.

    Same payload as /healthz, kept for compatibility with earlier setups
    that probe "/health" explicitly.
    """
    return {
        "status": "healthy",
        "model": miner_instance.model if miner_instance else "not_initialized",
        "model_loaded": miner_instance is not None,
        "backend": "openai-compatible",
        "model_type": miner_instance.model_type if miner_instance else None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    request: PredictRequest,
    http_request: Request,
):
    """
    Prediction endpoint with Bittensor protocol verification.

    Responsibilities:
      - Optionally log the client IP (proxy-safe) and request metadata.
      - Verify Bittensor dendrite-axon headers (unless dev mode is enabled).
      - Enforce that the request is addressed to *this* miner's hotkey.
      - Delegate to miner_instance.predict(request) to get the utterance
        completion from the OpenAI-compatible backend.

    Set MINER_DEV_MODE=1 (or true) to bypass verification for local testing.
    """
    if miner_instance is None:
        raise HTTPException(status_code=503, detail="Miner not initialized")

    # ----------------------------------------------------------------------
    # Proxy-safe client IP logging
    # ----------------------------------------------------------------------
    # We resolve the client IP *before* any Bittensor logic so that this
    # logging works both in dev mode and production mode.
    client_ip = get_client_ip(http_request)
    try:
        # If Bittensor headers are present, this is typically the validator's
        # external hotkey; we log at most a shortened prefix for hygiene.
        dendrite_hotkey_hdr = http_request.headers.get("bt_header_dendrite_hotkey")
        logger.info(
            "Incoming /predict from client_ip=%s dendrite_hotkey=%s index=%s step=%s",
            client_ip,
            (dendrite_hotkey_hdr[:16] + "...") if dendrite_hotkey_hdr else None,
            request.index,
            request.step,
        )
    except Exception:
        # Never let logging failures break the endpoint.
        logger.debug("Failed to log client IP for /predict", exc_info=True)
    # ----------------------------------------------------------------------

    # Check if dev mode is enabled (bypass verification).
    settings = get_settings()
    dev_mode = (
        getattr(settings, "MINER_DEV_MODE", False)
        or os.getenv("MINER_DEV_MODE", "0") in ("1", "true", "True")
    )

    if dev_mode:
        logger.info("🔓 Dev mode enabled - bypassing Bittensor verification")
        return await miner_instance.predict(request)

    # Extract Bittensor headers
    headers = http_request.headers

    # Required Bittensor protocol headers
    dendrite_hotkey = headers.get("bt_header_dendrite_hotkey")
    dendrite_nonce = headers.get("bt_header_dendrite_nonce")
    dendrite_signature = headers.get("bt_header_dendrite_signature")
    dendrite_uuid = headers.get("bt_header_dendrite_uuid")
    axon_hotkey = headers.get("bt_header_axon_hotkey")
    body_hash = headers.get("computed_body_hash", "")
    timeout_str = headers.get("timeout", "12.0")
    miner_hotkey = _get_miner_hotkey_ss58()

    # Check if this is a Bittensor protocol request
    is_bittensor_request = all(
        [
            dendrite_hotkey,
            dendrite_nonce,
            dendrite_signature,
            dendrite_uuid,
            axon_hotkey,
        ]
    )

    if is_bittensor_request:
        # Ensure the request is intended for this miner's hotkey.
        if miner_hotkey and axon_hotkey and axon_hotkey != miner_hotkey:
            logger.warning(
                "Rejecting request: target hotkey mismatch (expected %s, got %s)",
                miner_hotkey,
                axon_hotkey,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Request not addressed to this miner hotkey",
            )

        # Verify the request using Bittensor protocol.
        try:
            timeout = float(timeout_str)
        except ValueError:
            timeout = 12.0

        is_valid, error_msg = verify_bittensor_request(
            dendrite_hotkey=dendrite_hotkey,
            dendrite_nonce=dendrite_nonce,
            dendrite_signature=dendrite_signature,
            dendrite_uuid=dendrite_uuid,
            axon_hotkey=axon_hotkey,
            body_hash=body_hash,
            timeout=timeout,
        )

        if not is_valid:
            logger.warning(
                "Request verification failed from %s...: %s",
                dendrite_hotkey[:8] if dendrite_hotkey else "unknown",
                error_msg,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Request verification failed: {error_msg}",
            )

        logger.info(
            "✅ Verified request from validator: %s...",
            dendrite_hotkey[:8] if dendrite_hotkey else "unknown",
        )
    else:
        # Non-Bittensor request - only allowed in dev mode.
        if not dev_mode:
            logger.warning("Rejecting request without Bittensor headers (production mode)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bittensor protocol headers required",
            )

        logger.info("Processing request without Bittensor verification (dev mode)")

    # Process the prediction request via the OpenAI-compatible backend.
    return await miner_instance.predict(request)


async def main():
    """
    Main entry point for the miner server.

    Responsibilities:
      - Configure root logging.
      - Read axon port from settings.
      - Log mode (dev vs production).
      - Start Uvicorn with this FastAPI app.
    """
    # Setup logging for the entire process.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()
    axon_port = settings.MINER_AXON_PORT

    logger.info("=" * 60)
    logger.info("Starting Babelbit Miner Server (OpenAI-compatible backend)")
    logger.info("=" * 60)
    logger.info("")

    # Check dev mode
    dev_mode = (
        getattr(settings, "MINER_DEV_MODE", False)
        or os.getenv("MINER_DEV_MODE", "0") in ("1", "true", "True")
    )
    if dev_mode:
        logger.warning("🔓 DEV MODE ENABLED - Bittensor verification DISABLED")
        logger.warning("   This should ONLY be used for local testing!")
        logger.warning("   Set MINER_DEV_MODE=0 for production use.")
        logger.info("")
    else:
        logger.info("🔒 Bittensor verification enabled (production mode)")
        logger.info("")

    logger.info("⚠️  Make sure you've registered your axon first:")
    logger.info("   uv run python babelbit/miner/register_axon.py")
    logger.info("")

    import uvicorn

    # Start FastAPI server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=axon_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    logger.info("🚀 Miner serving predictions on port %d", axon_port)
    logger.info("   Press Ctrl+C to stop.")
    logger.info("")

    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down miner server...")


if __name__ == "__main__":
    asyncio.run(main())
