import test from "node:test";
import assert from "node:assert/strict";
import {
  modelFor,
  scenarioAvailable,
} from "../../src/features/models/availability.ts";

const model = (modelId, revision, available, seats = 4) => ({
  modelId,
  revision,
  displayName: modelId,
  version: revision,
  seats,
  available,
});
const catalog = (...models) => ({ models, defaultModels: [] });
const written = (modelId, revision) => ({ modelId, revision });
const key = (model) => (model ? `${model.modelId}@${model.revision}` : undefined);
// 画面がモデルを選ぶときのキー（modelKey と同じ形）。
const choice = (modelId, revision) => JSON.stringify([modelId, revision]);

const mixed = {
  models: [
    model("nano", "ppo17", true),
    model("prism", "v760", true),
    model("prism-3p", "u3-v300", true, 3),
    model("prism-3p", "u3-v355", true, 3),
  ],
  defaultModels: [written("prism", "v760"), written("prism-3p", "u3-v355")],
};

test("each player count starts on its own default model", () => {
  assert.equal(key(modelFor(mixed, 4, [])), "prism@v760");
  assert.equal(key(modelFor(mixed, 3, [])), "prism-3p@u3-v355");
});

test("a model the user chose wins for its player count only, the newest choice first", () => {
  const choices = [choice("prism-3p", "u3-v300"), choice("nano", "ppo17")];
  assert.equal(key(modelFor(mixed, 3, choices)), "prism-3p@u3-v300");
  assert.equal(key(modelFor(mixed, 4, choices)), "nano@ppo17");
  assert.equal(key(modelFor(mixed, 4, [choice("prism-3p", "u3-v300")])), "prism@v760");
});

test("unavailable choices and defaults fall through to the first usable model of that player count", () => {
  const down = {
    models: [
      model("nano", "ppo17", true),
      model("prism", "v760", false),
      model("prism-3p", "u3-v355", false, 3),
    ],
    defaultModels: [written("prism", "v760"), written("prism-3p", "u3-v355")],
  };
  assert.equal(key(modelFor(down, 4, [choice("prism", "v760")])), "nano@ppo17");
  assert.equal(modelFor(down, 3, []), undefined);
  assert.equal(key(modelFor(catalog(model("nano", "ppo17", true)), 4, [])), "nano@ppo17");
  assert.equal(modelFor(catalog(), 4, [choice("nano", "ppo17")]), undefined);
});

// Java は、解析を作ったバージョンが使えなければ、そのモデルの今のバージョン（models.json で最後のバージョン）でもしもに答える。
test("the what-if is offered when Java answers it: with the written revision or the model's current one", () => {
  const both = catalog(
    model("prism", "v700", true),
    model("prism", "v760", true),
  );
  assert.equal(scenarioAvailable(both, written("prism", "v700")), true);
  assert.equal(scenarioAvailable(both, written("prism", "v760")), true);

  const replaced = catalog(
    model("nano", "ppo17", true),
    model("prism", "v760", true),
  );
  assert.equal(scenarioAvailable(replaced, written("prism", "v700")), true);

  const changed = catalog(
    model("prism", "v700", false),
    model("prism", "v760", true),
  );
  assert.equal(scenarioAvailable(changed, written("prism", "v700")), true);
});

test("the what-if is not offered when neither the written revision nor the current one can run", () => {
  const currentDown = catalog(
    model("prism", "v700", true),
    model("prism", "v760", false),
  );
  assert.equal(scenarioAvailable(currentDown, written("prism", "v650")), false);
  assert.equal(scenarioAvailable(currentDown, written("prism", "v760")), false);
  assert.equal(scenarioAvailable(currentDown, written("prism", "v700")), true);

  const other = catalog(model("nano", "ppo17", true));
  assert.equal(scenarioAvailable(other, written("prism", "v700")), false);
  assert.equal(scenarioAvailable(catalog(), written("prism", "v700")), false);
});
