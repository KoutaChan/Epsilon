import type { ReviewRound, Snapshot } from "../../../api/types";

export interface TableFrame {
  snapshot: Snapshot;
  event: Snapshot;
  round: ReviewRound;
  viewSeat: number;
  reveal: boolean;
  animateDraw: boolean;
  layout: "desktop" | "portrait" | "landscape";
  labels: {
    remaining: string;
    loadingTiles: string;
    draw: string;
    round: string;
    winds: readonly string[];
  };
}

export interface TableRenderer {
  update(frame: TableFrame): void;
  dispose(): void;
}
