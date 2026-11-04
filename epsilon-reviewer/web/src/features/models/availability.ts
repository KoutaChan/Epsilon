import type {
  CatalogModel,
  ModelCatalog,
  ModelKey,
} from "../../../../protocol/model.ts";
import { modelKey } from "../../language/format.ts";

/** カタログで使えるモデル。key は modelKey の値。 */
export function availableModel(
  catalog: ModelCatalog,
  key: string,
): CatalogModel | undefined {
  return catalog.models.find(
    (model) => model.available && modelKey(model) === key,
  );
}

export function isAvailable(catalog: ModelCatalog, model: ModelKey): boolean {
  return availableModel(catalog, modelKey(model)) !== undefined;
}

/** カタログを読んだ後の選択: 前の選択 → 既定のモデル → なし（使えるものだけ）。 */
export function initialChoice(catalog: ModelCatalog, previous: string): string {
  if (availableModel(catalog, previous)) return previous;
  const fallback = catalog.defaultModel ? modelKey(catalog.defaultModel) : "";
  return availableModel(catalog, fallback) ? fallback : "";
}
