import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_embedding_module(monkeypatch):
    tracker = SimpleNamespace(gme_load_calls=0, text_load_calls=0, clients=[])

    class FakeGmeModel:
        def eval(self):
            return self

    class FakeTextModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    fake_modelscope = ModuleType("modelscope")
    fake_transformers = ModuleType("transformers")
    fake_ollama = ModuleType("ollama")
    fake_openai = ModuleType("openai")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            tracker.text_load_calls += 1
            return FakeTextModel()

    class FakeTransformerAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            tracker.gme_load_calls += 1
            return FakeGmeModel()

    class FakeOpenAIClient:
        def __init__(self, *args, **kwargs):
            self.closed = False
            tracker.clients.append(self)

        def close(self):
            self.closed = True

    fake_modelscope.AutoTokenizer = FakeAutoTokenizer
    fake_modelscope.AutoModel = FakeAutoModel
    fake_transformers.AutoModel = FakeTransformerAutoModel
    fake_ollama.embeddings = lambda **kwargs: {"embedding": [0.0, 1.0]}
    fake_openai.OpenAI = FakeOpenAIClient

    for name, module in {
        "modelscope": fake_modelscope,
        "transformers": fake_transformers,
        "ollama": fake_ollama,
        "openai": fake_openai,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[1] / "Core" / "provider" / "embedding.py"
    spec = importlib.util.spec_from_file_location("test_embedding_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, tracker


def test_gme_close_resets_singleton_and_allows_reinit(monkeypatch):
    embedding_mod, tracker = _load_embedding_module(monkeypatch)

    provider = embedding_mod.GmeEmbeddingProvider(model_name="fake-gme", device="cpu")

    assert tracker.gme_load_calls == 1
    assert provider._initialized is True

    provider.close()

    assert provider._initialized is False
    assert not hasattr(provider, "model")

    same_provider = embedding_mod.GmeEmbeddingProvider(model_name="fake-gme", device="cpu")

    assert same_provider is provider
    assert tracker.gme_load_calls == 2
    assert same_provider._initialized is True


def test_gme_close_instance_discards_singleton(monkeypatch):
    embedding_mod, tracker = _load_embedding_module(monkeypatch)

    provider = embedding_mod.GmeEmbeddingProvider(model_name="fake-gme", device="cpu")
    embedding_mod.GmeEmbeddingProvider.close_instance()

    assert embedding_mod.GmeEmbeddingProvider._instance is None

    new_provider = embedding_mod.GmeEmbeddingProvider(model_name="fake-gme", device="cpu")

    assert new_provider is not provider
    assert tracker.gme_load_calls == 2


@pytest.mark.parametrize("backend", ["local", "openai"])
def test_text_embedding_provider_close_releases_resources(monkeypatch, caplog, backend):
    embedding_mod, tracker = _load_embedding_module(monkeypatch)

    if backend == "local":
        provider = embedding_mod.TextEmbeddingProvider(
            model_name="fake-text",
            backend="local",
            device="cpu",
        )
        assert tracker.text_load_calls == 1
        assert hasattr(provider, "model")
        assert hasattr(provider, "tokenizer")
        client = None
    else:
        provider = embedding_mod.TextEmbeddingProvider(
            model_name="fake-text",
            backend="openai",
            api_key="test-key",
            api_base="http://example.test/v1",
        )
        client = tracker.clients[-1]

    with caplog.at_level(logging.WARNING):
        provider.close()

    assert "Calling close() on a singleton instance is discouraged" not in caplog.text
    if backend == "local":
        assert not hasattr(provider, "model")
        assert not hasattr(provider, "tokenizer")
    else:
        assert client.closed is True


def test_text_close_instance_warns_it_is_not_singleton(monkeypatch, caplog):
    embedding_mod, _ = _load_embedding_module(monkeypatch)

    with caplog.at_level(logging.WARNING):
        embedding_mod.TextEmbeddingProvider.close_instance()

    assert "TextEmbeddingProvider is not a singleton" in caplog.text