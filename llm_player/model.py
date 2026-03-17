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
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass


class ModelBackend:
    """Unified interface over Qwen 3B (local) or Gemini Flash (API).

    Auto-detects the available backend:
    - If RISK_MODEL_PATH is set and vllm is installed -> Qwen
    - If GOOGLE_API_KEY is set and google-genai is installed -> Gemini
    - Otherwise raises RuntimeError

    Args:
        backend: "qwen", "gemini", or "auto" (default).
        model_path: Path to local Qwen model weights. Overrides RISK_MODEL_PATH.
    """

    def __init__(self, backend: str = "auto",
                 model_path: Optional[str] = None):
        self.backend_type = None
        self._model = None

        if backend == "auto":
            # Try Qwen first, then Gemini
            qwen_path = model_path or os.environ.get("RISK_MODEL_PATH")
            if qwen_path and self._vllm_available():
                self._init_qwen(qwen_path)
            elif os.environ.get("GOOGLE_API_KEY") and self._genai_available():
                self._init_gemini()
            else:
                raise RuntimeError(
                    "No model backend available. Set RISK_MODEL_PATH "
                    "(with vllm installed) or GOOGLE_API_KEY "
                    "(with google-genai installed)."
                )
        elif backend == "qwen":
            path = model_path or os.environ.get("RISK_MODEL_PATH")
            if not path:
                raise RuntimeError("RISK_MODEL_PATH not set and no model_path provided.")
            self._init_qwen(path)
        elif backend == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'qwen', or 'gemini'.")

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
                 temperature: float = 0.7) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: The full prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            The model's text completion.
        """
        if self.backend_type == "qwen":
            return self._generate_qwen(prompt, max_tokens, temperature)
        elif self.backend_type == "gemini":
            return self._generate_gemini(prompt, max_tokens, temperature)
        else:
            raise RuntimeError("No backend initialized.")

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
        response = self._client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text


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
                 temperature: float = 0.7) -> str:
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
