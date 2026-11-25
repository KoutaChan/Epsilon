import type { HistoryList } from "@epsilon-reviewer/protocol/model";
import { api } from "../../api/platform";
import { useResource, type Resource } from "../../hooks/useResource";

export function useHistory(): Resource<HistoryList> {
  return useResource((signal) => api.history(signal), []);
}
