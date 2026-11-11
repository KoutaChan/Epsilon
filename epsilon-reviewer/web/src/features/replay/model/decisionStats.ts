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
  /** 価値を推定した判断の平均損失。そうした判断が無ければ null。 */
  readonly meanLoss: number | null;
}

// 同じ牌の手出しとツモ切りは、1 つの打牌として扱う。赤牌は別の手のままにする。
function choiceKey(row: CandidateRow): string | number {
  const { type, tile } = actionOf(row[0]);
  return type === "DAHAI" || type === "RIICHI_DAHAI"
    ? `${type}:${tile}`
    : row[0];
}

/** 席ごとに、対局全体で評価した判断の数・一致した数・悪手の数と、平均損失を返す。どれも候補の行から直接数える。 */
export function summarizePlayerDecisions(
  model: Pick<ReplayModel, "rounds" | "temperature">,
): PlayerDecisionSummary[] {
  const valued = model.temperature !== undefined;
  const seats = model.rounds[0]!.seats;
  const players = Array.from({ length: seats }, () => ({
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
      // まとめた打牌の評価は、価値なら良い方（小さい方）の損失とし、確率なら 2 つの合計とする。
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
        // まとめた手のどちらかを実際に選んでいれば、まとめた打牌を選んだとみなす。
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
