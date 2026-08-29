// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The picker panel is narrower than a typical window. Viewport min-[560px]
// column widths held empty meta slots open and clipped On Device names
// (github.com/unslothai/unsloth/issues/9965).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

const PICKERS = read(
  "../src/features/model-picker/components/model-selector/pickers.tsx",
);
const SELECTOR = read(
  "../src/features/model-picker/components/model-selector.tsx",
);
const CSS = read("../src/index.css");

test("row meta widths follow the picker panel, not the window", () => {
  assert.match(SELECTOR, /unsloth-model-selector-menu @container /);
  assert.match(PICKERS, /quant: "@min-\[560px\]:w-\[7\.2em\]"/);
  assert.match(PICKERS, /badge: "min-w-min @min-\[560px\]:w-\[24px\]"/);
  assert.equal(
    /(?<![@\w-])min-\[560px\]:w-/.test(PICKERS),
    false,
    "viewport min-[560px] widths must not remain on META_COLUMN",
  );
});

test("empty quant, badge, and size slots do not reserve a column", () => {
  assert.match(PICKERS, /alignMeta === "device" && quantChip \?/);
  assert.match(PICKERS, /aligned && hasBadgeContent \?/);
  assert.match(
    PICKERS,
    /alignMeta === "device" \|\| showSize\s*\n\s+\? parsed\.size !== undefined &&/,
  );
});

test("the picker panel is wide enough for local model names", () => {
  const start = CSS.indexOf("--picker-panel-w:");
  assert.ok(start >= 0);
  const block = CSS.slice(start, start + 280);
  assert.match(block, /max\(/);
  assert.match(block, /40rem/);
  assert.match(block, /--picker-control-w/);
});
