import { useCallback, useState } from "react";
import type {
  CatalogModel,
  ModelCatalog,
  ModelKey,
} from "@epsilon-reviewer/protocol/model";
import { api } from "../../api/platform";
import { useResource, type ResourceState } from "../../hooks/useResource";
import { modelFor, scenarioAvailable } from "./availability";

export interface Models {
  state: ResourceState<ModelCatalog>;
  /**
   * その人数の牌譜で選んでいるモデル（{@link modelFor}）。人数ごとに、利用者が選んだモデル、既定のモデルの順に決まる。カタログを
   * 読む前と、その人数の使えるモデルが無いときは undefined。
   */
  selected(seats: number): CatalogModel | undefined;
  /** モデルをキー（modelKey の値）で選ぶ。そのモデルの人数の牌譜で、次から選ばれる。カタログを読む前に選んでも残る。 */
  select(key: string): void;
  /** そのバージョンのモデルで作った解析結果の、もしもを解析できるか。カタログを読む前は false。 */
  scenarioAvailable(written: ModelKey): boolean;
  reload(): void;
}

export function useModels(): Models {
  const [choices, setChoices] = useState<readonly string[]>([]);
  const { state, reload } = useResource(
    async (signal) => api.models(signal),
    [],
  );
  const catalog = state.status === "ready" ? state.value : null;
  const selected = useCallback(
    (seats: number) =>
      catalog === null ? undefined : modelFor(catalog, seats, choices),
    [catalog, choices],
  );
  const select = useCallback(
    (key: string) =>
      setChoices((previous) => [key, ...previous.filter((k) => k !== key)]),
    [],
  );
  const available = useCallback(
    (written: ModelKey) =>
      catalog !== null && scenarioAvailable(catalog, written),
    [catalog],
  );
  return {
    state,
    selected,
    select,
    scenarioAvailable: available,
    reload,
  };
}
