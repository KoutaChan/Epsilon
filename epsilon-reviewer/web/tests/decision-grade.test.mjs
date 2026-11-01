import test from "node:test";
import assert from "node:assert/strict";
import {
  MISTAKE_LOSS,
  NEAR_LOSS,
  gradeLoss,
  gradeProbability,
  hasValueLoss,
} from "../src/features/replay/decisionGrade.ts";

test("loss grades use half-open bands above the best action", () => {
  for (const [loss, grade] of [
    [0, "best"],
    [1e-12, "near"],
    [NEAR_LOSS - 1e-9, "near"],
    [NEAR_LOSS, "inaccuracy"],
    [MISTAKE_LOSS - 1e-9, "inaccuracy"],
    [MISTAKE_LOSS, "mistake"],
    [1, "mistake"],
  ])
    assert.equal(gradeLoss(loss), grade, String(loss));
});

test("probability grades keep tied tops and the strict five-percent bound", () => {
  assert.equal(gradeProbability(0.3, 0.1 + 0.2), "top");
  assert.equal(gradeProbability(0.04, 0.04), "top");
  assert.equal(gradeProbability(0.05, 0.95), "other");
  assert.equal(gradeProbability(0.05 - 1e-10, 0.95), "bad");
});

test("value steps are recognised from their candidates", () => {
  const candidate = (valueLoss) => ({ valueLoss });
  assert.equal(hasValueLoss([]), false);
  assert.equal(hasValueLoss([candidate(null), candidate(null)]), false);
  assert.equal(hasValueLoss([candidate(0), candidate(0.01)]), true);
});
