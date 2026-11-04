import test from "node:test";
import assert from "node:assert/strict";
import { INITIAL_NAVIGATION, navigate } from "../../src/app/navigation.ts";

const record = {
  recordId: "4e7c1c3a-0b39-4c43-8b8e-2f1d5a3c9e10",
  record: { source: "tenhou", names: ["A", "B", "C", "D"], roundCount: 8 },
};
const model = {
  modelId: "prism",
  revision: "v551",
  displayName: "Prism",
  version: "v2",
};
const document = {
  format: 6,
  resultId: "82b7a497-06a6-42a1-8700-8b4b65e83273",
};
const entry = {
  resultId: document.resultId,
  createdAt: "",
  record: record.record,
  model,
};

const run = (...actions) => actions.reduce(navigate, INITIAL_NAVIGATION);
const confirm = (seat = 0) => ({ name: "confirm", record, seat });
const progress = (seat) => ({
  name: "progress",
  record,
  seat,
  jobId: "job",
  model,
});

test("an import opens the confirmation with the first seat", () => {
  assert.deepEqual(run({ type: "imported", record }).screen, confirm(0));
});

test("the chosen seat travels from confirmation through progress to the review", () => {
  const started = run(
    { type: "imported", record },
    { type: "seatChosen", seat: 2 },
    { type: "started", started: { jobId: "job", model } },
  );
  assert.deepEqual(started.screen, progress(2));
  assert.deepEqual(navigate(started, { type: "completed", document }).screen, {
    name: "review",
    document,
    seat: 2,
  });
});

test("back goes from progress to confirmation, and from confirmation or review home", () => {
  const started = run(
    { type: "imported", record },
    { type: "seatChosen", seat: 1 },
    { type: "started", started: { jobId: "job", model } },
  );
  const confirmation = navigate(started, { type: "back" });
  assert.deepEqual(confirmation.screen, confirm(1));
  assert.deepEqual(navigate(confirmation, { type: "back" }).screen, {
    name: "home",
  });
  const review = run({ type: "opened", document });
  assert.deepEqual(review.screen, { name: "review", document, seat: 0 });
  assert.deepEqual(navigate(review, { type: "back" }).screen, { name: "home" });
});

test("stale actions leave the screen alone", () => {
  const home = run();
  assert.equal(navigate(home, { type: "completed", document }), home);
  assert.equal(navigate(home, { type: "seatChosen", seat: 3 }), home);
  assert.equal(
    navigate(home, { type: "started", started: { jobId: "job", model } }),
    home,
  );
});

test("a new results folder returns home from confirmation and progress only", () => {
  const confirmation = run({ type: "imported", record });
  assert.deepEqual(navigate(confirmation, { type: "folderChanged" }).screen, {
    name: "home",
  });
  const started = navigate(confirmation, {
    type: "started",
    started: { jobId: "job", model },
  });
  assert.deepEqual(navigate(started, { type: "folderChanged" }).screen, {
    name: "home",
  });
  const review = run({ type: "opened", document });
  assert.equal(navigate(review, { type: "folderChanged" }), review);
});

test("modals open and close without changing the screen, and the review closes them", () => {
  const withDelete = run(
    { type: "imported", record },
    { type: "modalOpened", modal: { name: "delete", entry } },
  );
  assert.deepEqual(withDelete.screen, confirm(0));
  assert.deepEqual(withDelete.modal, { name: "delete", entry });
  assert.equal(navigate(withDelete, { type: "modalClosed" }).modal, null);
  const storage = run({ type: "modalOpened", modal: { name: "storage" } });
  assert.equal(navigate(storage, { type: "opened", document }).modal, null);
});
