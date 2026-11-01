import type { Candidate, ReviewRound } from "../../api/types";
import {
  gradeLoss,
  gradeProbability,
  hasValueLoss,
} from "./decisionGrade.ts";

export interface PlayerDecisionSummary {
  reviewed: number;
  matched: number;
  badMoves: number;
  /** 価値を推定した判断での平均損失。該当する判断がなければ null。 */
  meanLoss: number | null;
}

// 同じ牌の手出しとツモ切りは一つの打牌として扱う。赤牌は別の選択のまま残す。
function choiceKey(candidate: Candidate): string | number {
  return candidate.type === "DAHAI" || candidate.type === "RIICHI_DAHAI"
    ? `${candidate.type}:${candidate.tiles[0]}`
    : candidate.actionId;
}

export function summarizePlayerDecisions(
  rounds: readonly Pick<ReviewRound, "steps">[],
): PlayerDecisionSummary[] {
  const players = Array.from({ length: 4 }, () => ({
    reviewed: 0,
    matched: 0,
    badMoves: 0,
    valued: 0,
    totalLoss: 0,
  }));
  for (const round of rounds) {
    for (const step of round.steps) {
      if (!step.decision) continue;
      const valued = hasValueLoss(step.candidates);
      // 価値は同じ打牌の良い方の損失、確率は合算を打牌の評価とする。
      const scores = new Map<string | number, number>();
      let chosenKey: string | number | undefined;
      for (const candidate of step.candidates) {
        const key = choiceKey(candidate),
          previous = scores.get(key);
        scores.set(
          key,
          valued
            ? Math.min(previous ?? Infinity, candidate.valueLoss!)
            : (previous ?? 0) + candidate.probability,
        );
        // 同じ打牌のどちらかが実行されていれば、集約した打牌を実際の選択とする。
        if (candidate.chosen) chosenKey = key;
      }
      if (chosenKey === undefined || scores.size < 2) continue;
      const chosen = scores.get(chosenKey)!;
      const stats = players[step.actor];
      stats.reviewed++;
      if (valued) {
        const grade = gradeLoss(chosen);
        stats.valued++;
        stats.totalLoss += chosen;
        if (grade === "best") stats.matched++;
        else if (grade === "mistake") stats.badMoves++;
      } else {
        const grade = gradeProbability(chosen, Math.max(...scores.values()));
        if (grade === "top") stats.matched++;
        else if (grade === "bad") stats.badMoves++;
      }
    }
  }
  return players.map(({ valued, totalLoss, ...counts }) => ({
    ...counts,
    meanLoss: valued === 0 ? null : totalLoss / valued,
  }));
}
