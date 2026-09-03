// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Regression coverage for https://github.com/unslothai/unsloth/issues/10227

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");

function read(relativePath: string): string {
  return readFileSync(path.join(ROOT, relativePath), "utf8");
}

test("successful UI loads mirror remembered launch settings for API auto-switch", () => {
  const runtimeSource = read(
    "src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  assert.match(runtimeSource, /mirrorLaunchConfigForApiLoad\(/);
  assert.match(runtimeSource, /rememberedLaunch\.remembered/);
});

test("hub load passes the config identity used for remembered settings", () => {
  const hubSource = read("src/features/hub/hub-page.tsx");
  assert.match(hubSource, /configId: configIdentity/);
});

test("model config save still mirrors directly from the settings page", () => {
  const configPage = read(
    "src/features/model-picker/components/model-config-page.tsx",
  );
  assert.match(configPage, /syncModelOverride\(/);
});
