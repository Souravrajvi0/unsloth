// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { mirrorLaunchConfigForApiLoad } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const MODEL = "Myric/KAT-Coder-V2.5-Dev-MTP-APEX-GGUF";
const VARIANT = "KAT-Coder-V2.5-Dev-APEX-dynamic-v2";

test("mirrorLaunchConfigForApiLoad pushes non-default remembered settings", async () => {
  const puts: unknown[] = [];
  setAuthFetchHandler(async (_url, init) => {
    puts.push(JSON.parse(String(init?.body ?? "{}")));
    return new Response(JSON.stringify({ overrides: {} }), { status: 200 });
  });

  mirrorLaunchConfigForApiLoad({
    modelId: MODEL,
    ggufVariant: VARIANT,
    config: {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 65536,
      kvCacheDtype: "q8_0",
    },
    eligible: true,
  });

  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(puts.length, 1);
  const body = puts[0] as {
    model_id: string;
    custom_context_length?: number;
    kv_cache_dtype?: string;
  };
  assert.equal(body.model_id, `${MODEL}:${VARIANT}`);
  assert.equal(body.custom_context_length, 65536);
  assert.equal(body.kv_cache_dtype, "q8_0");
});

test("mirrorLaunchConfigForApiLoad skips ineligible and default configs", async () => {
  const puts: unknown[] = [];
  setAuthFetchHandler(async (_url, init) => {
    puts.push(JSON.parse(String(init?.body ?? "{}")));
    return new Response(JSON.stringify({ overrides: {} }), { status: 200 });
  });

  mirrorLaunchConfigForApiLoad({
    modelId: MODEL,
    ggufVariant: VARIANT,
    config: DEFAULT_PER_MODEL_CONFIG,
    eligible: true,
  });
  mirrorLaunchConfigForApiLoad({
    modelId: MODEL,
    ggufVariant: VARIANT,
    config: {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 4096,
    },
    eligible: false,
  });

  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(puts.length, 0);
});
