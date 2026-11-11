import test from "node:test";
import assert from "node:assert/strict";
import { ApiError } from "../../src/api/errors.ts";
import { jobEnded } from "../../src/features/analysis/jobEnd.ts";

test("a job has ended without a result once it failed or was cancelled", () => {
  assert.equal(jobEnded("failed", null), true);
  assert.equal(jobEnded("cancelled", null), true);
  for (const status of ["queued", "running", "saving", "completed"])
    assert.equal(jobEnded(status, null), false);
});

// ジョブはメモリにしか無いので、解析する側が起動し直すと無くなる。その後は、読み直しても取り消しても job_not_found になる。
test("a job that no longer exists has ended, whatever status was read last", () => {
  const gone = new ApiError("job_not_found", "Job not found.", 404);
  assert.equal(jobEnded("queued", gone), true);
  assert.equal(jobEnded("running", gone), true);
});

test("a job that could not be read for another reason may still be running", () => {
  for (const code of ["connection_failed", "engine_failed", "invalid_response"])
    assert.equal(jobEnded("running", new ApiError(code, "")), false);
  assert.equal(jobEnded("running", new Error("job_not_found")), false);
});
