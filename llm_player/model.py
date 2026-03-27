"""Model backend abstraction for Qwen 3B and Gemini Flash fallback.

Provides a unified generate() interface for the decision nodes.
Auto-detects available backend: local Qwen model via vLLM/transformers
(env var RISK_MODEL_PATH) or Gemini Flash via API (env var GOOGLE_API_KEY).

Also provides MockModelBackend for unit testing without a real model.

Usage:
    backend = ModelBackend()  # auto-detects
    completion = backend.generate(prompt, max_tokens=512)
"""

import os
import time
import logging
from typing import Optional

LOG = logging.getLogger("llm_player.model")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass


class ModelBackend:
    """Unified interface over Qwen (local) or Gemini Flash (API).

    Auto-detects the available backend:
    - If RISK_MODEL_PATH points to a PEFT adapter and peft is installed -> PEFT
    - If RISK_MODEL_PATH is set and vllm is installed -> Qwen via vLLM
    - If GOOGLE_API_KEY is set and google-genai is installed -> Gemini
    - Otherwise raises RuntimeError

    Args:
        backend: "peft", "qwen", "gemini", or "auto" (default).
        model_path: Path to local model weights or PEFT adapter. Overrides RISK_MODEL_PATH.
    """

    def __init__(self, backend: str = "auto",
                 model_path: Optional[str] = None):
        self.backend_type = None
        self._model = None
        self._tokenizer = None
        self.call_count = 0
        self.call_counts_by_caller = {}  # tracks calls per node
        self.call_log = []  # records prompt/response/caller for each call

        if backend == "auto":
            qwen_path = model_path or os.environ.get("RISK_MODEL_PATH")
            if qwen_path and self._is_peft_adapter(qwen_path) and self._peft_available():
                self._init_peft(qwen_path)
            elif qwen_path and not self._is_peft_adapter(qwen_path) and self._transformers_available():
                # Raw HuggingFace model name or local dir without adapter
                self._init_transformers(qwen_path)
            elif qwen_path and self._vllm_available():
                self._init_qwen(qwen_path)
            elif os.environ.get("GOOGLE_API_KEY") and self._genai_available():
                self._init_gemini()
            else:
                raise RuntimeError(
                    "No model backend available. Set RISK_MODEL_PATH "
                    "(with vllm, peft, or transformers installed) or GOOGLE_API_KEY "
                    "(with google-genai installed)."
                )
        elif backend == "peft":
            path = model_path or os.environ.get("RISK_MODEL_PATH")
            if not path:
                raise RuntimeError("RISK_MODEL_PATH not set and no model_path provided.")
            self._init_peft(path)
        elif backend == "transformers":
            path = model_path or os.environ.get("RISK_MODEL_PATH")
            if not path:
                raise RuntimeError("RISK_MODEL_PATH not set and no model_path provided.")
            self._init_transformers(path)
        elif backend == "qwen":
            path = model_path or os.environ.get("RISK_MODEL_PATH")
            if not path:
                raise RuntimeError("RISK_MODEL_PATH not set and no model_path provided.")
            self._init_qwen(path)
        elif backend == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'peft', 'transformers', 'qwen', or 'gemini'.")

    @staticmethod
    def _vllm_available() -> bool:
        try:
            import vllm  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _genai_available() -> bool:
        try:
            from google import genai  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _peft_available() -> bool:
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _transformers_available() -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _is_peft_adapter(path: str) -> bool:
        """Check if path contains adapter_config.json (a PEFT LoRA adapter)."""
        return os.path.isdir(path) and os.path.exists(
            os.path.join(path, "adapter_config.json")
        )

    def _init_peft(self, adapter_path: str):
        """Load base model + LoRA adapter via PEFT for inference."""
        import json as _json
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path) as f:
            adapter_cfg = _json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]

        load_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        LOG.info("Loading base model: %s", base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, **load_kwargs
        )
        LOG.info("Applying LoRA adapter: %s", adapter_path)
        self._model = PeftModel.from_pretrained(base_model, adapter_path)
        self._model.eval()

        # Try loading tokenizer from adapter dir first, fall back to base model
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                adapter_path, trust_remote_code=True
            )
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True
            )

        self.backend_type = "peft"
        LOG.info("PEFT backend ready (base: %s)", base_model_name)

    def _init_transformers(self, model_name: str):
        """Load a plain HuggingFace model via transformers (no LoRA adapter)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        LOG.info("Loading model via transformers: %s", model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.backend_type = "peft"  # reuse _generate_peft (same logic)
        LOG.info("Transformers backend ready: %s", model_name)

    def _init_qwen(self, model_path: str):
        from vllm import LLM
        self._model = LLM(model=model_path, max_model_len=4096)
        self.backend_type = "qwen"

    def _init_gemini(self):
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
        self._client = genai.Client(api_key=api_key)
        self.backend_type = "gemini"

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.7, caller: str = "") -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: The full prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            caller: Label for tracking which node made the call.

        Returns:
            The model's text completion.
        """
        self.call_count += 1
        if caller:
            self.call_counts_by_caller[caller] = self.call_counts_by_caller.get(caller, 0) + 1
        if self.backend_type == "peft":
            result = self._generate_peft(prompt, max_tokens, temperature)
        elif self.backend_type == "qwen":
            result = self._generate_qwen(prompt, max_tokens, temperature)
        elif self.backend_type == "gemini":
            result = self._generate_gemini(prompt, max_tokens, temperature)
        else:
            raise RuntimeError("No backend initialized.")
        self.call_log.append({"prompt": prompt, "response": result, "caller": caller})
        return result

    def _generate_peft(self, prompt: str, max_tokens: int,
                       temperature: float) -> str:
        """Generate using PEFT model with chat template applied."""
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_qwen(self, prompt: str, max_tokens: int,
                       temperature: float) -> str:
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=max_tokens,
                                temperature=temperature)
        outputs = self._model.generate([prompt], params)
        return outputs[0].outputs[0].text

    def _generate_gemini(self, prompt: str, max_tokens: int,
                         temperature: float) -> str:
        from google.genai import types
        max_retries = 20
        for attempt in range(max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                return response.text
            except Exception as e:
                err = str(e).lower()
                retryable = (
                    "429" in err or "rate" in err
                    or "resource" in err or "quota" in err
                    or "503" in err or "unavailable" in err
                    or "500" in err or "server" in err
                )
                if attempt < max_retries and retryable:
                    wait = 90  # wait for per-minute quota to reset
                    LOG.warning("API error (attempt %d/%d), waiting %ds: %s",
                                attempt + 1, max_retries, wait,
                                str(e)[:100])
                    time.sleep(wait)
                else:
                    raise


class MockModelBackend:
    """Test double that returns predetermined responses.

    For unit tests that need a model backend without API calls.

    Args:
        responses: List of strings to return in order. Cycles if exhausted.
            If None, returns a default valid JSON response based on prompt content.
    """

    def __init__(self, responses: Optional[list] = None):
        self._responses = responses or []
        self._call_count = 0
        self.backend_type = "mock"
        self.call_log = []

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.7, caller: str = "") -> str:
        """Return next predetermined response, or auto-detect from prompt."""
        self.call_log.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
            self._call_count += 1
            return response

        # Auto-detect response type from prompt content
        return self._auto_response(prompt)

    def _auto_response(self, prompt: str) -> str:
        """Generate a valid default response based on prompt type."""
        prompt_lower = prompt.lower()
        if "reinforcement" in prompt_lower:
            return '```json\n{"reinforcements": {}}\n```'
        elif "attack" in prompt_lower:
            return '```json\n{"attacks": []}\n```'
        elif "movement" in prompt_lower:
            return '```json\n{"movement": null}\n```'
        else:
            return "I recommend focusing on consolidating your position."
