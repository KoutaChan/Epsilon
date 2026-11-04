import { api } from "./api/platform";
import { OwnerApp } from "./app/OwnerApp";
import { SharedReview } from "./features/sharing/SharedReview";

/** Web で ?share= があれば共有の閲覧（値が空でも）。デスクトップは ?share を読まない。 */
export function App() {
  const shared =
    api.platform === "web"
      ? new URLSearchParams(window.location.search).get("share")
      : null;
  return shared === null ? <OwnerApp /> : <SharedReview resultId={shared} />;
}
