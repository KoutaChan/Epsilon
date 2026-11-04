import { useCallback, useState } from "react";
import type {
  CatalogModel,
  ModelCatalog,
  ModelKey,
} from "../../../../protocol/model.ts";
import { api } from "../../api/platform";
import { useResource, type ResourceState } from "../../hooks/useResource";
import { availableModel, initialChoice, isAvailable } from "./availability";

export interface Models {
  state: ResourceState<ModelCatalog>;
  /** 選んだモデル。カタログを読む前と、選んだモデルが使えない時は undefined。 */
  selected: CatalogModel | undefined;
  /** modelKey の値で選ぶ。カタログを読む前に選べば、読んだ後も使える限り残る。 */
  select(key: string): void;
  isAvailable(model: ModelKey): boolean;
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
    (model: ModelKey) => catalog !== null && isAvailable(catalog, model),
    [catalog],
  );
  return {
    state,
    selected: catalog ? availableModel(catalog, choice) : undefined,
    select,
    isAvailable: available,
    reload,
  };
}
