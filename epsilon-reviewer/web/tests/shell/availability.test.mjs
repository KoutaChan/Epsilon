import test from "node:test";
import assert from "node:assert/strict";
import { scenarioAvailable } from "../../src/features/models/availability.ts";

const model = (modelId, revision, available) => ({
  modelId,
  revision,
  displayName: modelId,
  version: revision,
  seats: 4,
  available,
});
const catalog = (...models) => ({ models, defaultModel: null });
const written = (modelId, revision) => ({ modelId, revision });

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
