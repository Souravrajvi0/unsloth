# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Non-string tool arguments must not crash the chat turn.

Local models emit lists/objects/numbers for query/url/code/command.
``value.strip()`` then raises AttributeError and kills the generation.
"""

import ast
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "core" / "inference" / "tools.py"


def _load_tool_text_arg():
    source = TOOLS.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tool_text_arg"
    )
    namespace = {}
    module = ast.Module(body = [func], type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(TOOLS), "exec"), namespace)
    return namespace["_tool_text_arg"]


def test_tool_text_arg_keeps_strings():
    coerce = _load_tool_text_arg()
    assert coerce("hello") == "hello"
    assert coerce("") == ""


def test_tool_text_arg_drops_non_strings():
    coerce = _load_tool_text_arg()
    for value in (None, False, True, 0, 1, [], {}, ["print(1)"], {"q": "x"}):
        assert coerce(value) == "", value


def test_execute_tool_coerces_query_url_code_and_command():
    source = TOOLS.read_text(encoding = "utf-8")
    assert '_tool_text_arg(arguments.get("query"' in source
    assert '_tool_text_arg(arguments.get("url")' in source
    assert '_tool_text_arg(arguments.get("code"' in source
    assert '_tool_text_arg(arguments.get("command"' in source
    # Must not stringify the value: that would search/write the repr.
    assert 'str(arguments.get("query"' not in source
    assert 'str(arguments.get("url"' not in source
