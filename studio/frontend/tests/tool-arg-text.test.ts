// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { toolArgText } from "../src/components/assistant-ui/tool-arg-text.ts";

test("toolArgText keeps real strings", () => {
  assert.equal(toolArgText("hello"), "hello");
  assert.equal(toolArgText(""), "");
});

test("toolArgText does not throw on the shapes local models emit", () => {
  for (const value of [undefined, null, 0, 1, false, true, [], {}, ["q"], { q: "x" }]) {
    assert.equal(toolArgText(value), "");
  }
});
