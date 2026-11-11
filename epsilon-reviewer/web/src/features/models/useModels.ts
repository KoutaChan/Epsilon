import { useCallback, useState } from "react";
import type {
  CatalogModel,
  ModelCatalog,
  ModelKey,
} from "../../../../protocol/model.ts";
import { api } from "../../api/platform";
import { useResource, type ResourceState } from "../../hooks/useResource";
import {
  availableModel,
  initialChoice,
  scenarioAvailable,
} from "./availability";

export interface Models {
  state: ResourceState<ModelCatalog>;
  /** 選んだモデル。カタログを読む前と、選んだモデルが使えないときは undefined。 */
  selected: CatalogModel | undefined;
  /** モデルをキー（modelKey の値）で選ぶ。カタログを読む前に選んでも、読んだ後にそのモデルが使えれば選択は残る。 */
  select(key: string): void;
  /** そのバージョンのモデルで作った解析結果の、もしもを解析できるか。カタログを読む前は false。 */
  scenarioAvailable(written: ModelKey): boolean;
  reload(): void;
}

export function useModels(): Models {
  const [choice, select] = useState("");
  const { state, reload } = useResource(async (signal) => {
    const catalog = await api.models(signal);
    select((previous) => initialChoice(catalog, previous));
    return catalog;
  }, []);
  const catalog = state.status === "ready" ? state.value : null;
  const available = useCallback(
    (written: ModelKey) =>
      catalog !== null && scenarioAvailable(catalog, written),
    [catalog],
  );
  return {
    state,
    selected: catalog ? availableModel(catalog, choice) : undefined,
    select,
    scenarioAvailable: available,
    reload,
  };
}
