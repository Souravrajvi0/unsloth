# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reproduction for https://github.com/unslothai/unsloth/issues/10227

Custom per-model launch settings (context length, KV cache dtype) saved in the
Desktop UI can be ignored when the same model is auto-loaded via the OpenAI API
(e.g. GitHub Copilot), because API auto-switch reads server-side model overrides,
not browser localStorage alone.
"""

import pytest

from utils import openai_auto_switch_settings as settings

from test_openai_auto_switch import (
    _FakeBackend,
    _LoadRecorder,
    _mock_override_store,
    _run_hook,
    _wire,
)


# Model shape from the issue report (non-standard quant label after the colon).
ISSUE_MODEL = (
    "Myric/KAT-Coder-V2.5-Dev-MTP-APEX-GGUF:KAT-Coder-V2.5-Dev-APEX-dynamic-v2"
)
ISSUE_REPO = "Myric/KAT-Coder-V2.5-Dev-MTP-APEX-GGUF"
ISSUE_VARIANT = "KAT-Coder-V2.5-Dev-APEX-dynamic-v2"


@pytest.fixture
def override_store(monkeypatch):
    """In-memory override store (empty unless a test writes to it)."""
    return _mock_override_store(monkeypatch)


def test_issue_10227_api_autoload_uses_defaults_without_server_override(
    monkeypatch,
    override_store,
):
    """Reproduces #10227: remembered UI settings that never reached the server."""
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = (ISSUE_REPO, ISSUE_VARIANT, ISSUE_REPO),
        backend = backend,
        recorder = rec,
    )

    # Browser localStorage would hold custom settings here; the API path never reads it.
    assert settings.get_model_override(ISSUE_REPO) == {}
    assert settings.get_model_override(f"{ISSUE_REPO}:{ISSUE_VARIANT}") == {}

    _run_hook(ISSUE_MODEL)

    assert len(rec.calls) == 1
    req = rec.calls[0]
    # Defaults: no stored override -> bare LoadRequest fields (context 0 = model default).
    assert req.max_seq_length == 0
    assert req.cache_type_kv is None
    assert req.gguf_variant == ISSUE_VARIANT


def test_issue_10227_api_autoload_honors_server_override_when_present(
    monkeypatch,
    override_store,
):
    """Control: once mirrored to the server, auto-switch applies the saved launch config."""
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = (ISSUE_REPO, ISSUE_VARIANT, ISSUE_REPO),
        backend = backend,
        recorder = rec,
    )

    settings.set_model_override(
        f"{ISSUE_REPO}:{ISSUE_VARIANT}",
        max_seq_length = 65536,
        kv_cache_dtype = "q8_0",
    )

    _run_hook(ISSUE_MODEL)

    assert len(rec.calls) == 1
    req = rec.calls[0]
    assert req.max_seq_length == 65536
    assert req.cache_type_kv == "q8_0"


def test_issue_10227_resolve_override_matches_variant_qualified_key(override_store):
    """The issue's variant label must resolve the same way UI + API both key it."""
    settings.set_model_override(
        f"{ISSUE_REPO}:{ISSUE_VARIANT}",
        max_seq_length = 49152,
        kv_cache_dtype = "q8_0",
    )
    key, override = settings.resolve_override_for_load(
        ISSUE_REPO,
        ISSUE_REPO,
        ISSUE_VARIANT,
    )
    assert key == f"{ISSUE_REPO}:{ISSUE_VARIANT}"
    assert override["max_seq_length"] == 49152
    assert override["kv_cache_dtype"] == "q8_0"
