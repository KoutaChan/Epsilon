import type { HistoryList } from "../../../../protocol/model.ts";
import { api } from "../../api/platform";
import { useResource, type Resource } from "../../hooks/useResource";

export function useHistory(): Resource<HistoryList> {
  return useResource((signal) => api.history(signal), []);
}
