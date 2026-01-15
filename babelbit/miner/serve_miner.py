"""
Simple FastAPI-based miner server for Babelbit subnet.
Serves predictions via HTTP endpoint that validators can call directly.

Note: Run register_axon.py first to register your miner on-chain,
then run this script to serve predictions.
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from traceback import format_exc
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Header, status
from pydantic import BaseModel
import uvicorn
from substrateinterface import Keypair

from babelbit.miner.model_loader import load_model_and_tokenizer
from babelbit.miner.utils import verify_bittensor_request
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from babelbit.utils.settings import get_settings

logger = logging.getLogger(__name__)

# Simple in-process cache for tokenized static prompt prefixes
_PROMPT_CACHE: dict[str, torch.Tensor] = {}
_MINER_HOTKEY_SS58: Optional[str] = None


# -----------------------------------------------------------------------------
# Post-processing helpers (kept intentionally lightweight)
# -----------------------------------------------------------------------------
# These helpers enforce the same constraints your training/harness uses:
#   - Truncate at the utterance-level EOF marker (if configured)
#   - Ensure a single-sentence prediction (token-based heuristic)
#   - Join prefix + completion without introducing spurious punctuation/spacing
#
# Rationale:
# Validators score against *utterance-level* ground truth. Even when the model
# learns to stop, decoding can still include extra tail tokens. These rules
# provide a deterministic guardrail consistent with your finetune harness.
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "inc.", "ltd.", "corp.", "co.",
    "e.g.", "i.e.", "etc.", "vs.",
    "u.s.", "u.k.", "no.", "st.", "mt.",
}

PROMPT_TEMPLATE = """
<|system|>
You predict the full utterance from a partial prefix. Output only the completion.
<|user|>
Context:
{context}
Prefix:
{prefix}
<|assistant|>
"""


def _truncate_at_eof_token(text: str, eof_token: str) -> str:
    """Truncate `text` at the first occurrence of `eof_token` (literal substring)."""
    if not eof_token:
        return text
    idx = text.find(eof_token)
    if idx < 0:
        return text
    return text[:idx].rstrip()


def _is_sentence_end_token(tok: str) -> bool:
    """
    Token-based sentence boundary heuristic:
      - strip trailing quotes/brackets
      - ignore common abbreviations
      - treat tokens ending with . ! ? as a sentence terminator
    """
    if not tok:
        return False
    stripped = tok.rstrip('"\')]}')
    if not stripped:
        return False
    low = stripped.lower()
    if low in _ABBREVIATIONS:
        return False
    return stripped.endswith((".", "!", "?"))


def truncate_to_one_sentence_tokenwise(text: str) -> str:
    """Keep only the first completed sentence in token space (split by whitespace)."""
    text = (text or "").strip()
    if not text:
        return text
    tokens = text.split()
    out: list[str] = []
    for t in tokens:
        out.append(t)
        if _is_sentence_end_token(t):
            break
    return " ".join(out).strip()


def _join_prefix_completion(prefix: str, completion: str) -> str:
    """
    Join prefix + completion without adding unwanted spaces/punctuation.

    - If completion begins with whitespace, concatenate directly.
    - If completion begins with punctuation, concatenate directly.
    - Otherwise add exactly one space between prefix and completion.
    """
    prefix = (prefix or "").strip()
    completion = completion or ""
    if not prefix:
        return completion.strip()
    if not completion:
        return prefix

    if completion.startswith(prefix):
        return completion

    if completion.strip().startswith(prefix):
        return completion.strip()

    if completion[:1].isspace():
        return (prefix + completion).strip()

    if completion[0] in {".", ",", ":", ";", "!", "?", ")", "]", "}", "'"}:
        return (prefix + completion).strip()

    return (prefix + " " + completion).strip()


def _get_miner_hotkey_ss58() -> Optional[str]:
    """Load and cache this miner's hotkey SS58 address."""
    global _MINER_HOTKEY_SS58
    if _MINER_HOTKEY_SS58:
        return _MINER_HOTKEY_SS58
    try:
        settings = get_settings()
        keypair = load_hotkey_keypair(settings.BITTENSOR_WALLET_COLD, settings.BITTENSOR_WALLET_HOT)
        _MINER_HOTKEY_SS58 = keypair.ss58_address
        return _MINER_HOTKEY_SS58
    except Exception as e:
        logger.warning(f"Unable to load miner hotkey for request verification: {e}")
        return None


class BBUtteranceEvaluation(BaseModel):
    """Evaluation result for utterance prediction."""
    lexical_similarity: float = 0.0
    semantic_similarity: float = 0.0
    earliness: float = 0.0
    u_step: float = 0.0


class PredictRequest(BaseModel):
    """Request schema matching chute template."""
    index: str  # UUID session identifier
    step: int
    prefix: str
    context: str = ""
    done: bool = False
    ground_truth: str | None = None
    prediction: str = ""
    evaluation: BBUtteranceEvaluation | None = None


class PredictResponse(BaseModel):
    """Simple response schema expected by validator."""
    prediction: str


class BabelbitMiner:
    """Miner that serves predictions using a Hugging Face model."""
    
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        adapter_model_id: Optional[str] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the miner with a model.
        
        Args:
            model_id: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-hf")
            revision: Model revision/branch to use
            cache_dir: Directory for model cache
            device: Device to load model on ("cuda" or "cpu")
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.adapter_model_id = adapter_model_id
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Determine device and dtype
        self.device = self._pick_device() if device == "cuda" else torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Model and tokenizer loaded on demand
        self._model = None
        self._tokenizer = None
        self._model_lock = asyncio.Lock()
        self._model_moved = False
        
        logger.info(f"Initialized BabelbitMiner with model: {model_id}")
        logger.info(f"Adapter model ID: {adapter_model_id}")
        logger.info(f"Target device: {self.device}, dtype: {self.dtype}")
    
    def _pick_device(self) -> torch.device:
        """Select best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _get_env_int(self, name: str, default: int) -> int:
        """Get integer from environment variable."""
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default
    
    def _get_env_float(self, name: str, default: float) -> float:
        """Get float from environment variable."""
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default
    
    def _prepare_inputs(self, prompt: str) -> torch.Tensor:
        """Tokenize prompt with caching of static system+instruction part."""
        try:
            # Fallback: no split
            inputs = self._tokenizer.encode(prompt, return_tensors="pt")
            if inputs.numel() == 0:
                raise ValueError("Empty tokenization result")
            return inputs.to(self.device)
            
        except Exception as e:
            logger.error(f"Error in _prepare_inputs: {str(e)}")
            # Emergency fallback - create a simple tensor with EOS token
            if hasattr(self._tokenizer, 'eos_token_id') and self._tokenizer.eos_token_id is not None:
                fallback_tensor = torch.tensor([[self._tokenizer.eos_token_id]], dtype=torch.long)
            else:
                fallback_tensor = torch.tensor([[1]], dtype=torch.long)
            return fallback_tensor.to(self.device)
    
    async def load(self):
        """Load model and tokenizer (called once at startup)."""
        async with self._model_lock:
            if self._model is None:
                logger.info(f"Loading model {self.model_id}...")
                logger.info(f"Adapter model ID: {self.adapter_model_id}")

                self._model, self._tokenizer = await asyncio.to_thread(
                    load_model_and_tokenizer,
                    model_id=self.model_id,
                    revision=self.revision,
                    cache_dir=self.cache_dir,
                    adapter_model_id=self.adapter_model_id,
                    device=self.device,
                    load_in_8bit=self.load_in_8bit,
                    load_in_4bit=self.load_in_4bit,
                )
                logger.info(f"Model loaded successfully on {self.device}")
    
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Generate prediction for the given prefix and context.
        
        Args:
            request: Prediction request with prefix and context
            
        Returns:
            PredictResponse with generated text (just the prediction string)
        """
        try:
            # Ensure model is loaded
            if self._model is None:
                await self.load()
            
            if not request.prefix:
                logger.warning("No prefix provided, returning empty prediction")
                return PredictResponse(prediction="")
            
            logger.info(f"Generating prediction for prefix: '{request.prefix}'")
            logger.info(f"Using context: '{request.context}'")

            # ---------------------------------------------------------------------
            # PROMPT FORMAT (MUST MATCH TRAINING/HARNESS)
            # ---------------------------------------------------------------------
            # Training/harness builds prompts as:
            #   - prompt = prefix                                (if no context)
            #   - prompt = context + context_separator + prefix   (if context)
            #
            # Therefore, we must NOT wrap the request into a chat template or
            # instruction-style prompt here; doing so creates inference-time
            # distribution shift relative to training and tends to reduce SN59
            # validator score.
            #
            # The separator must match your training wrapper (default: " EOF ").
            # Keep it configurable via env var for operational flexibility.
            sep = os.getenv("MINER_CONTEXT_SEPARATOR", " EOF ")
            prompt = PROMPT_TEMPLATE.format(context=request.context, prefix=request.prefix)

            # Transformers edge case: empty prompt can yield empty input_ids and
            # break generation. Match harness behavior by forcing a safe token.
            if len(prompt.strip()) == 0:
                prompt = self._tokenizer.eos_token or ""

            # Move model to device lazily (only first call)
            if not self._model_moved:
                try:
                    self._model.to(self.device)
                    self._model.eval()
                    logger.info(f"Model moved to {self.device}")
                    self._model_moved = True
                except Exception as e:
                    logger.error(f"Error moving model to device: {str(e)}")
                    # Fallback to CPU
                    self.device = torch.device("cpu")
                    self._model.to(self.device)
                    self._model.eval()
                    logger.info("Fell back to CPU device")
                    self._model_moved = True
            
            # Tokenize with caching
            try:
                inputs = await asyncio.to_thread(self._prepare_inputs, prompt)
                
                # Validate input tensor
                if inputs.dim() != 2 or inputs.size(0) != 1:
                    raise ValueError(f"Invalid input tensor shape: {inputs.shape}")
                
                vocab_size = getattr(self._tokenizer, 'vocab_size', 50000)
                if torch.any(inputs >= vocab_size) or torch.any(inputs < 0):
                    raise ValueError("Input contains invalid token IDs")
                    
            except Exception as e:
                logger.error(f"Error preparing inputs: {str(e)}")
                # Create safe fallback input
                fallback_text = request.prefix[:50] if request.prefix else "Hello"
                inputs = self._tokenizer.encode(
                    fallback_text, return_tensors="pt", max_length=512, truncation=True
                )
                inputs = inputs.to(self.device)
            
            # Get generation parameters from environment
            max_new_tokens = self._get_env_int("CHUTE_MAX_NEW_TOKENS", 24)
            temperature = self._get_env_float("CHUTE_TEMPERATURE", 0.7)
            top_p = self._get_env_float("CHUTE_TOP_P", 0.95)
            top_k = self._get_env_int("CHUTE_TOP_K", 50)
            do_sample = os.getenv("CHUTE_DO_SAMPLE", "1") not in ("0", "false", "False")
            
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            
            # Check for early exit if prefix already ends with EOS
            if self._tokenizer.eos_token and request.prefix.strip().endswith(self._tokenizer.eos_token):
                logger.warning("Prefix already contains EOS token, returning as-is")
                return PredictResponse(prediction=request.prefix.strip())
            
            # Run generation with safety checks
            with torch.no_grad():
                try:
                    # Ensure input length is within model limits
                    max_pos = getattr(self._model.config, "max_position_embeddings", None)
                    if max_pos and inputs.size(1) > max_pos - 1:
                        # Truncate from the left if too long
                        inputs = inputs[:, -(max_pos - 1):]
                        logger.warning(f"Truncated input to {inputs.size(1)} tokens to fit model limit {max_pos}")
                    
                    def _generate_on_device():
                        return self._model.generate(inputs, **gen_kwargs)

                    outputs = await asyncio.to_thread(_generate_on_device)

                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logger.error("CUDA error during generation: %s", e)
                        logger.info("Retrying generation on CPU.")
                        # Move to CPU and retry once
                        inputs_cpu = inputs.cpu()
                        self._model.cpu()
                        self.device = torch.device("cpu")
                        with torch.no_grad():
                            outputs = self._model.generate(inputs_cpu, **gen_kwargs)
                    else:
                        raise

            # Decode only the NEW tokens after the prompt (more reliable than string slicing).
            # This avoids edge cases where generated_text does not start with the exact prompt
            # due to tokenizer normalization.
            prompt_len = inputs.size(1)

            # `outputs` is typically shape [1, seq_len]. Normalize to 1D token ids.
            out_ids = outputs[0] if outputs.ndim == 2 else outputs
            if out_ids.ndim != 1:
                raise ValueError(f"Unexpected output tensor shape: {out_ids.shape}")

            if prompt_len > out_ids.shape[0]:
                # Fallback: decode everything (should be rare).
                completion = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            else:
                completion = self._tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)

            completion = completion.lstrip("\\n\\r")

            # With a train-faithful raw prompt, the decoded new tokens are already the completion.
            prediction_tail = completion.strip()

            # ---------------------------------------------------------------------
            # Post-process and assemble final utterance prediction (prefix + completion)
            # ---------------------------------------------------------------------
            # 1) Join safely (no spurious spaces before punctuation).
            full_prediction = _join_prefix_completion(request.prefix, prediction_tail)

            # 2) Truncate at utterance EOF token.
            #    Your training wrapper appends the literal token (including spaces):
            #      --utterance_eof_token " <EOFUTR> "
            #    Do NOT .strip() this environment variable; stripping would change the
            #    literal marker and truncation would fail.
            utterance_eof_token = os.getenv("MINER_UTTERANCE_EOF_TOKEN", " <EOFUTR> ")
            if utterance_eof_token:
                full_prediction = _truncate_at_eof_token(full_prediction, utterance_eof_token)

            # 3) Enforce single-sentence output (default ON).
            truncate_one_sentence_env = os.getenv("MINER_TRUNCATE_ONE_SENTENCE", "1")
            truncate_one_sentence = truncate_one_sentence_env not in ("0", "false", "False")
            if truncate_one_sentence:
                full_prediction = truncate_to_one_sentence_tokenwise(full_prediction)

            # If still empty, fall back to the prefix (never invent punctuation-only tails).
            if not full_prediction or full_prediction.strip() == "":
                full_prediction = (request.prefix or "").strip()

            logger.info(f"Generated: {full_prediction[:100]}...")
            
            return PredictResponse(prediction=full_prediction)
            
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.error(format_exc())
            # Return empty prediction on error
            return PredictResponse(prediction="")


# Global miner instance
miner_instance: Optional[BabelbitMiner] = None


async def startup():
    """FastAPI startup event handler."""
    global miner_instance
    
    settings = get_settings()
    
    # Get model configuration
    model_id = settings.MINER_MODEL_ID
    revision = getattr(settings, 'MINER_MODEL_REVISION', None)
    adapter_model_id = getattr(settings, 'ADAPTER_MODEL_ID', None)
    cache_dir = settings.BABELBIT_CACHE_DIR / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Quantization settings
    load_in_8bit = getattr(settings, 'MINER_LOAD_IN_8BIT', False)
    load_in_4bit = getattr(settings, 'MINER_LOAD_IN_4BIT', False)
    device = getattr(settings, 'MINER_DEVICE', 'cuda')
    
    logger.info(f"Model: {model_id}")
    logger.info(f"Revision: {revision or 'main'}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Quantization: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Create and load miner
    miner_instance = BabelbitMiner(
        model_id=model_id,
        revision=revision,
        adapter_model_id=adapter_model_id,
        cache_dir=cache_dir,
        device=device,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    
    logger.info("Loading model...")
    try:
        await miner_instance.load()
        logger.info("✅ Model loaded successfully")
        logger.info("")
    except Exception as e:
        logger.error("❌ Failed to load model!")
        logger.error(f"   Error: {e}")
        logger.error("")
        logger.error("Common fixes:")
        logger.error("  1. For gated models (Llama, etc): Set HF_TOKEN environment variable")
        logger.error("     export HF_TOKEN=your_huggingface_token")
        logger.error("  2. Check model ID is correct and you have access")
        logger.error("  3. Ensure you have enough disk space and RAM/VRAM")
        raise


# Create FastAPI app
app = FastAPI(title="Babelbit Miner", on_startup=[startup])


@app.get("/healthz")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": miner_instance.model_id if miner_instance else "not_loaded",
        "model_loaded": miner_instance is not None and miner_instance._model is not None,
    }


@app.get("/health")
async def health_alt():
    """Alternative health check endpoint."""
    return {
        "status": "healthy",
        "model": miner_instance.model_id if miner_instance else "not_loaded",
        "model_loaded": miner_instance is not None and miner_instance._model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    request: PredictRequest,
    http_request: Request,
):
    """
    Prediction endpoint with Bittensor protocol verification.
    
    Validates incoming requests from validators using cryptographic signatures
    and nonce-based replay attack prevention.
    
    Set MINER_DEV_MODE=1 to bypass verification for local testing.
    """
    if miner_instance is None:
        raise HTTPException(status_code=503, detail="Miner not initialized")
    
    # Check if dev mode is enabled (bypass verification)
    settings = get_settings()
    dev_mode = getattr(settings, "MINER_DEV_MODE", False) or os.getenv("MINER_DEV_MODE", "0") in ("1", "true", "True")
    
    if dev_mode:
        logger.info("🔓 Dev mode enabled - bypassing Bittensor verification")
        return await miner_instance.predict(request)
    
    # Extract Bittensor headers
    headers = http_request.headers
    
    # Get required Bittensor protocol headers
    dendrite_hotkey = headers.get("bt_header_dendrite_hotkey")
    dendrite_nonce = headers.get("bt_header_dendrite_nonce")
    dendrite_signature = headers.get("bt_header_dendrite_signature")
    dendrite_uuid = headers.get("bt_header_dendrite_uuid")
    axon_hotkey = headers.get("bt_header_axon_hotkey")
    body_hash = headers.get("computed_body_hash", "")
    timeout_str = headers.get("timeout", "12.0")
    miner_hotkey = _get_miner_hotkey_ss58()
    
    # Check if this is a Bittensor protocol request
    is_bittensor_request = all([
        dendrite_hotkey,
        dendrite_nonce,
        dendrite_signature,
        dendrite_uuid,
        axon_hotkey,
    ])
    
    if is_bittensor_request:
        # Ensure the request is intended for this miner's hotkey
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

        # Verify the request using Bittensor protocol
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
            logger.warning(f"Request verification failed from {dendrite_hotkey[:8]}...: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Request verification failed: {error_msg}"
            )
        
        logger.info(f"✅ Verified request from validator: {dendrite_hotkey[:8]}...")
    else:
        # Non-Bittensor request - check dev mode
        settings = get_settings()
        dev_mode = getattr(settings, "MINER_DEV_MODE", False) or os.getenv("MINER_DEV_MODE", "0") in ("1", "true", "True")
        
        if not dev_mode:
            # In production mode, reject requests without Bittensor headers
            logger.warning("Rejecting request without Bittensor headers (production mode)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bittensor protocol headers required"
            )
        
        # Allow non-Bittensor requests only in dev mode
        logger.info("Processing request without Bittensor verification (dev mode)")
    
    # Process the prediction request
    return await miner_instance.predict(request)


async def main():
    """Main entry point for the miner server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    settings = get_settings()
    axon_port = settings.MINER_AXON_PORT
    
    logger.info("=" * 60)
    logger.info("Starting Babelbit Miner Server")
    logger.info("=" * 60)
    logger.info("")
    
    # Check dev mode
    dev_mode = getattr(settings, "MINER_DEV_MODE", False) or os.getenv("MINER_DEV_MODE", "0") in ("1", "true", "True")
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
    
    # Start FastAPI server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=axon_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    
    logger.info(f"🚀 Miner serving predictions on port {axon_port}")
    logger.info("   Press Ctrl+C to stop.")
    logger.info("")
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down miner server...")


if __name__ == "__main__":
    asyncio.run(main())
