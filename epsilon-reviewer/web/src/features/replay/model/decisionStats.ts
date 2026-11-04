import {
  actionOf,
  candidateProbabilities,
  type CandidateRow,
} from "../../../../../protocol/result.ts";
import { gradeLoss, gradeProbability } from "./decisionGrade.ts";
import type { ReplayModel } from "./types.ts";

export interface PlayerDecisionSummary {
  readonly reviewed: number;
  readonly matched: number;
  readonly badMoves: number;
  /** 価値を推定した判断での平均損失。該当する判断がなければ null。 */
  readonly meanLoss: number | null;
}

// 同じ牌の手出しとツモ切りは一つの打牌として扱う。赤牌は別の選択のまま残す。
function choiceKey(row: CandidateRow): string | number {
  const { type, tile } = actionOf(row[0]);
  return type === "DAHAI" || type === "RIICHI_DAHAI"
    ? `${type}:${tile}`
    : row[0];
}

/** 席ごとの対局全体の一致率・悪手率・平均損失。候補の行を直接数える。 */
export function summarizePlayerDecisions(
  model: Pick<ReplayModel, "rounds" | "temperature">,
): PlayerDecisionSummary[] {
  const valued = model.temperature !== undefined;
  const players = Array.from({ length: 4 }, () => ({
    reviewed: 0,
    matched: 0,
    badMoves: 0,
    valued: 0,
    totalLoss: 0,
  }));
  for (const round of model.rounds) {
    for (const step of round.steps) {
      if (!step.decision) continue;
      const { set } = step;
      // 価値は同じ打牌の良い方の損失、確率は合算を打牌の評価とする。
      const probabilities = valued
        ? null
        : candidateProbabilities(set, undefined);
      const scores = new Map<string | number, number>();
      let chosenKey: string | number | undefined;
      set.candidates.forEach((row, index) => {
        const key = choiceKey(row);
        const previous = scores.get(key);
        const value = probabilities === null ? row[1] : probabilities[index]!;
        scores.set(
          key,
          valued
            ? Math.min(previous ?? Infinity, value)
            : (previous ?? 0) + value,
        );
        // 同じ打牌のどちらかが実行されていれば、集約した打牌を実際の選択とする。
        if (index === set.chosen) chosenKey = key;
      });
      if (chosenKey === undefined || scores.size < 2) continue;
      const chosen = scores.get(chosenKey)!;
      const stats = players[step.actor]!;
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
  return players.map(({ valued: count, totalLoss, ...counts }) => ({
    ...counts,
    meanLoss: count === 0 ? null : totalLoss / count,
  }));
}
