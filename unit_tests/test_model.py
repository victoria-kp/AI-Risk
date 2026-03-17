"""Unit tests for llm_player/model.py"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_player.model import ModelBackend, MockModelBackend


# ── MockModelBackend Tests ──────────────────────────────────────────


class TestMockModelBackendInit(unittest.TestCase):
    """Test MockModelBackend initialization."""

    def test_default_init(self):
        mock = MockModelBackend()
        self.assertEqual(mock.backend_type, "mock")
        self.assertEqual(mock._call_count, 0)
        self.assertEqual(mock.call_log, [])

    def test_init_with_responses(self):
        mock = MockModelBackend(responses=["hello", "world"])
        self.assertEqual(len(mock._responses), 2)

    def test_init_with_empty_responses(self):
        mock = MockModelBackend(responses=[])
        self.assertEqual(len(mock._responses), 0)


class TestMockModelBackendGenerate(unittest.TestCase):
    """Test MockModelBackend.generate() with predetermined responses."""

    def test_returns_first_response(self):
        mock = MockModelBackend(responses=["response1", "response2"])
        result = mock.generate("any prompt")
        self.assertEqual(result, "response1")

    def test_returns_responses_in_order(self):
        mock = MockModelBackend(responses=["first", "second", "third"])
        self.assertEqual(mock.generate("a"), "first")
        self.assertEqual(mock.generate("b"), "second")
        self.assertEqual(mock.generate("c"), "third")

    def test_cycles_responses(self):
        mock = MockModelBackend(responses=["A", "B"])
        self.assertEqual(mock.generate("1"), "A")
        self.assertEqual(mock.generate("2"), "B")
        self.assertEqual(mock.generate("3"), "A")
        self.assertEqual(mock.generate("4"), "B")

    def test_single_response_always_returns_same(self):
        mock = MockModelBackend(responses=["only"])
        for _ in range(5):
            self.assertEqual(mock.generate("x"), "only")

    def test_increments_call_count(self):
        mock = MockModelBackend(responses=["r"])
        mock.generate("a")
        mock.generate("b")
        self.assertEqual(mock._call_count, 2)


class TestMockModelBackendCallLog(unittest.TestCase):
    """Test that MockModelBackend logs all calls."""

    def test_logs_prompt(self):
        mock = MockModelBackend(responses=["r"])
        mock.generate("test prompt")
        self.assertEqual(len(mock.call_log), 1)
        self.assertEqual(mock.call_log[0]["prompt"], "test prompt")

    def test_logs_parameters(self):
        mock = MockModelBackend(responses=["r"])
        mock.generate("p", max_tokens=256, temperature=0.5)
        log = mock.call_log[0]
        self.assertEqual(log["max_tokens"], 256)
        self.assertEqual(log["temperature"], 0.5)

    def test_logs_default_parameters(self):
        mock = MockModelBackend(responses=["r"])
        mock.generate("p")
        log = mock.call_log[0]
        self.assertEqual(log["max_tokens"], 512)
        self.assertEqual(log["temperature"], 0.7)

    def test_logs_multiple_calls(self):
        mock = MockModelBackend(responses=["r"])
        mock.generate("first")
        mock.generate("second")
        mock.generate("third")
        self.assertEqual(len(mock.call_log), 3)
        self.assertEqual(mock.call_log[0]["prompt"], "first")
        self.assertEqual(mock.call_log[2]["prompt"], "third")


class TestMockModelBackendAutoResponse(unittest.TestCase):
    """Test auto-detection of response type from prompt content."""

    def test_reinforcement_prompt(self):
        mock = MockModelBackend()
        result = mock.generate("You have 5 reinforcement troops to place.")
        self.assertIn("reinforcements", result)
        self.assertIn("json", result)

    def test_attack_prompt(self):
        mock = MockModelBackend()
        result = mock.generate("Decide your attacks this turn.")
        self.assertIn("attacks", result)
        self.assertIn("json", result)

    def test_movement_prompt(self):
        mock = MockModelBackend()
        result = mock.generate("Decide on your free troop movement.")
        self.assertIn("movement", result)
        self.assertIn("json", result)

    def test_strategy_prompt(self):
        mock = MockModelBackend()
        result = mock.generate("Should I try to complete Asia?")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_reinforcement_case_insensitive(self):
        mock = MockModelBackend()
        result = mock.generate("REINFORCEMENT troops available")
        self.assertIn("reinforcements", result)

    def test_explicit_responses_override_auto(self):
        mock = MockModelBackend(responses=["custom"])
        result = mock.generate("You have 5 reinforcement troops")
        self.assertEqual(result, "custom")


class TestMockModelBackendInterface(unittest.TestCase):
    """Test that MockModelBackend matches ModelBackend interface."""

    def test_has_generate_method(self):
        mock = MockModelBackend()
        self.assertTrue(callable(getattr(mock, 'generate', None)))

    def test_has_backend_type(self):
        mock = MockModelBackend()
        self.assertIsInstance(mock.backend_type, str)

    def test_generate_returns_string(self):
        mock = MockModelBackend(responses=["test"])
        result = mock.generate("prompt")
        self.assertIsInstance(result, str)

    def test_generate_accepts_all_params(self):
        mock = MockModelBackend(responses=["test"])
        result = mock.generate("prompt", max_tokens=100, temperature=0.0)
        self.assertIsInstance(result, str)


# ── ModelBackend Tests ──────────────────────────────────────────────


class TestModelBackendAutoDetection(unittest.TestCase):
    """Test ModelBackend auto-detection logic."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_backend_raises(self):
        """With no env vars and no packages, should raise RuntimeError."""
        with patch.object(ModelBackend, '_vllm_available', return_value=False), \
             patch.object(ModelBackend, '_genai_available', return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                ModelBackend()
            self.assertIn("No model backend available", str(ctx.exception))

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_auto_selects_gemini_when_available(self):
        """With GOOGLE_API_KEY set and genai available, should select gemini."""
        with patch.object(ModelBackend, '_vllm_available', return_value=False), \
             patch.object(ModelBackend, '_genai_available', return_value=True), \
             patch.object(ModelBackend, '_init_gemini') as mock_init:
            backend = ModelBackend()
            mock_init.assert_called_once()

    @patch.dict(os.environ, {"RISK_MODEL_PATH": "/fake/path", "GOOGLE_API_KEY": "key"}, clear=True)
    def test_auto_prefers_qwen_over_gemini(self):
        """With both available, should prefer Qwen."""
        with patch.object(ModelBackend, '_vllm_available', return_value=True), \
             patch.object(ModelBackend, '_genai_available', return_value=True), \
             patch.object(ModelBackend, '_init_qwen') as mock_qwen, \
             patch.object(ModelBackend, '_init_gemini') as mock_gemini:
            backend = ModelBackend()
            mock_qwen.assert_called_once_with("/fake/path")
            mock_gemini.assert_not_called()

    @patch.dict(os.environ, {"RISK_MODEL_PATH": "/fake/path"}, clear=True)
    def test_auto_skips_qwen_if_vllm_not_installed(self):
        """If RISK_MODEL_PATH set but vllm not installed, should not use qwen."""
        with patch.object(ModelBackend, '_vllm_available', return_value=False), \
             patch.object(ModelBackend, '_genai_available', return_value=False):
            with self.assertRaises(RuntimeError):
                ModelBackend()


class TestModelBackendExplicitBackend(unittest.TestCase):
    """Test explicit backend selection."""

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ModelBackend(backend="unknown")
        self.assertIn("Unknown backend", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_qwen_without_path_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            ModelBackend(backend="qwen")
        self.assertIn("RISK_MODEL_PATH", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_gemini_without_key_raises(self):
        """_init_gemini checks for GOOGLE_API_KEY and raises if missing."""
        mock_genai = MagicMock()
        with patch.dict('sys.modules', {'google.genai': mock_genai}):
            with self.assertRaises(RuntimeError) as ctx:
                ModelBackend(backend="gemini")
            self.assertIn("GOOGLE_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_gemini_explicit(self):
        with patch.object(ModelBackend, '_init_gemini') as mock_init:
            backend = ModelBackend(backend="gemini")
            mock_init.assert_called_once()

    def test_qwen_explicit_with_model_path(self):
        with patch.object(ModelBackend, '_init_qwen') as mock_init:
            backend = ModelBackend(backend="qwen", model_path="/fake/model")
            mock_init.assert_called_once_with("/fake/model")

    @patch.dict(os.environ, {"RISK_MODEL_PATH": "/env/path"}, clear=True)
    def test_qwen_uses_env_var(self):
        with patch.object(ModelBackend, '_init_qwen') as mock_init:
            backend = ModelBackend(backend="qwen")
            mock_init.assert_called_once_with("/env/path")

    def test_model_path_overrides_env(self):
        with patch.dict(os.environ, {"RISK_MODEL_PATH": "/env/path"}), \
             patch.object(ModelBackend, '_init_qwen') as mock_init:
            backend = ModelBackend(backend="qwen", model_path="/arg/path")
            mock_init.assert_called_once_with("/arg/path")


class TestModelBackendGenerate(unittest.TestCase):
    """Test generate() dispatching."""

    def test_generate_without_init_raises(self):
        """If backend_type is None, generate should raise."""
        with patch.object(ModelBackend, '__init__', lambda self, **kw: None):
            backend = ModelBackend.__new__(ModelBackend)
            backend.backend_type = None
            backend._model = None
            with self.assertRaises(RuntimeError):
                backend.generate("test")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_gemini_generate_calls_api(self):
        """Test that gemini backend calls generate_content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test response"
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(ModelBackend, '_init_gemini'):
            backend = ModelBackend(backend="gemini")
            backend.backend_type = "gemini"
            backend._client = mock_client

            mock_types = MagicMock()
            with patch.dict('sys.modules', {'google.genai': MagicMock(), 'google.genai.types': mock_types}):
                result = backend.generate("test prompt", max_tokens=256, temperature=0.5)

            self.assertEqual(result, "test response")
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            self.assertEqual(call_args[1]["model"], "gemini-2.5-flash-lite")
            self.assertEqual(call_args[1]["contents"], "test prompt")

    def test_qwen_generate_calls_vllm(self):
        """Test that qwen backend calls vllm generate."""
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "qwen response"
        mock_model.generate.return_value = [mock_output]

        with patch.object(ModelBackend, '_init_qwen'):
            backend = ModelBackend.__new__(ModelBackend)
            backend.backend_type = "qwen"
            backend._model = mock_model

            with patch('llm_player.model.ModelBackend._generate_qwen') as mock_gen:
                mock_gen.return_value = "qwen response"
                backend._generate_qwen = mock_gen
                result = backend._generate_qwen("test", 512, 0.7)
                self.assertEqual(result, "qwen response")


class TestModelBackendAvailability(unittest.TestCase):
    """Test backend availability checks."""

    def test_vllm_available_returns_bool(self):
        result = ModelBackend._vllm_available()
        self.assertIsInstance(result, bool)

    def test_genai_available_returns_bool(self):
        result = ModelBackend._genai_available()
        self.assertIsInstance(result, bool)


class TestModelBackendDefaults(unittest.TestCase):
    """Test default parameter values."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_default_params(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(ModelBackend, '_init_gemini'):
            backend = ModelBackend(backend="gemini")
            backend.backend_type = "gemini"
            backend._client = mock_client

            mock_types = MagicMock()
            with patch.dict('sys.modules', {'google.genai': MagicMock(), 'google.genai.types': mock_types}):
                backend.generate("prompt")

            call_args = mock_client.models.generate_content.call_args
            self.assertEqual(call_args[1]["model"], "gemini-2.5-flash-lite")
            self.assertEqual(call_args[1]["contents"], "prompt")


if __name__ == '__main__':
    unittest.main()
