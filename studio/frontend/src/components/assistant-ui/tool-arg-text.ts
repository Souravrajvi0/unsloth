// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Coerce a tool-call argument to the string the card already expected.

Local models often emit arrays or objects for `query` / `url` / `code` /
`command`. Calling `.trim()` or `.split()` on those throws and unmounts the
chat thread.
*/
export function toolArgText(value: unknown): string {
  return typeof value === "string" ? value : "";
}
