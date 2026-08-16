// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { wireToolCallIdForReplay } from "../src/features/chat/tool-call-id.ts";

test("wireToolCallIdForReplay prefers an explicit wire id", () => {
  assert.equal(
    wireToolCallIdForReplay("call_1:550e8400-e29b-41d4-a716-446655440000", "call_1"),
    "call_1",
  );
});

test("wireToolCallIdForReplay strips a minted uuid suffix", () => {
  assert.equal(
    wireToolCallIdForReplay("call_0:550e8400-e29b-41d4-a716-446655440000"),
    "call_0",
  );
});

test("wireToolCallIdForReplay leaves confirmation-scoped ids intact", () => {
  const confirmationId = "thread-scope:approval-7";
  assert.equal(wireToolCallIdForReplay(confirmationId), confirmationId);
});

test("wireToolCallIdForReplay leaves plain provider ids intact", () => {
  assert.equal(wireToolCallIdForReplay("call_abc123"), "call_abc123");
});
