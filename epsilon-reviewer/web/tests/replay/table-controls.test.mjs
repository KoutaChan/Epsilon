import test from "node:test";
import assert from "node:assert/strict";
import { DESKTOP_BAR, PHONE_BARS } from "./table-fixtures.mjs";

test("the control sizes include the what-if button on every layout", () => {
  assert.deepEqual(DESKTOP_BAR, { units: 742, height: 62 });
  assert.deepEqual(PHONE_BARS, {
    landscape: { width: 308, height: 44, right: 8, top: null },
    portrait: { width: 176, height: 88, right: 10, top: 16 },
  });
});
