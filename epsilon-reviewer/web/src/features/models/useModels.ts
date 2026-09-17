import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { ModelCatalog } from "../../api/types";
import { modelKey } from "../../language/format";

type CatalogState =
  | { status: "loading" }
  | { status: "error"; error: unknown }
  | { status: "ready"; catalog: ModelCatalog };
export function useModels(connection: number) {
  const [state, setState] = useState<CatalogState>({ status: "loading" });
  const [selectedKey, select] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading" });
    api
      .models(controller.signal)
      .then((catalog) => {
        if (controller.signal.aborted) return;
        setState({ status: "ready", catalog });
        select((previous) => {
          if (
            catalog.models.some(
              (model) =>
                modelKey(model) === previous &&
                model.availability === "available",
            )
          )
            return previous;
          const defaultKey = catalog.defaultModel
            ? modelKey(catalog.defaultModel)
            : "";
          return catalog.models.some(
            (model) =>
              modelKey(model) === defaultKey &&
              model.availability === "available",
          )
            ? defaultKey
            : "";
        });
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setState({ status: "error", error });
      });
    return () => controller.abort();
  }, [connection, revision]);
  const selected =
    state.status === "ready"
      ? state.catalog.models.find(
          (model) =>
            modelKey(model) === selectedKey &&
            model.availability === "available",
        )
      : undefined;
  return {
    state,
    selectedKey,
    selected,
    select,
    refresh: () => setRevision((value) => value + 1),
  };
}
export type Models = ReturnType<typeof useModels>;
