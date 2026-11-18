import type {
  CatalogModel,
  ModelCatalog,
  ModelKey,
} from "@epsilon-reviewer/protocol/model";
import { modelKey } from "../../language/format.ts";

/** カタログから、使えるモデルをキー（modelKey の値）で探す。使えるモデルが見つからなければ undefined を返す。 */
export function availableModel(
  catalog: ModelCatalog,
  key: string,
): CatalogModel | undefined {
  return catalog.models.find(
    (model) => model.available && modelKey(model) === key,
  );
}

/**
 * 解析結果のもしもを解析できるか。Java は、解析を作ったバージョンが使えればそれで、使えなければそのモデルの今のバージョン
 * （一覧で最後のバージョン）で答える。だから、そのどちらかが使えるときに true を返す。
 *
 * @param written 解析を作ったモデルのバージョン
 */
export function scenarioAvailable(
  catalog: ModelCatalog,
  written: ModelKey,
): boolean {
  if (availableModel(catalog, modelKey(written))) return true;
  const current = catalog.models
    .filter((model) => model.modelId === written.modelId)
    .at(-1);
  return current?.available === true;
}

/**
 * その人数の牌譜で選ぶモデル。使える、その人数のモデルの中から、利用者が選んだモデル（新しく選んだ順）、その人数の既定のモデル
 * （defaultModels の順）、一覧で最初のモデルの順に探す。どれも無ければ undefined を返す。
 *
 * @param choices 利用者が選んだモデルのキー（modelKey の値、新しく選んだ順）
 */
export function modelFor(
  catalog: ModelCatalog,
  seats: number,
  choices: readonly string[],
): CatalogModel | undefined {
  const usable = (key: string) => {
    const model = availableModel(catalog, key);
    return model?.seats === seats ? model : undefined;
  };
  for (const key of choices) {
    const model = usable(key);
    if (model) return model;
  }
  for (const key of catalog.defaultModels) {
    const model = usable(modelKey(key));
    if (model) return model;
  }
  return catalog.models.find((model) => model.available && model.seats === seats);
}
