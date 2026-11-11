import { api } from "./api/platform";
import { OwnerApp } from "./app/OwnerApp";
import { SharedReview } from "./features/sharing/SharedReview";

/** Web 版では、URL に ?share= があれば共有された解析結果を表示する（値が空でも表示する）。デスクトップ版は ?share を見ない。 */
export function App() {
  const shared =
    api.platform === "web"
      ? new URLSearchParams(window.location.search).get("share")
      : null;
  return shared === null ? <OwnerApp /> : <SharedReview resultId={shared} />;
}
