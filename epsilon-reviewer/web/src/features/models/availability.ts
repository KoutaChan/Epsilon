import type {
  CatalogModel,
  ModelCatalog,
  ModelKey,
} from "../../../../protocol/model.ts";
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

/** カタログを読んだ後に選ぶモデルのキー。前に選んでいたモデル、既定のモデルの順に見て、使える方を返す。どちらも使えなければ空文字列を返す。 */
export function initialChoice(catalog: ModelCatalog, previous: string): string {
  if (availableModel(catalog, previous)) return previous;
  const fallback = catalog.defaultModel ? modelKey(catalog.defaultModel) : "";
  return availableModel(catalog, fallback) ? fallback : "";
}
