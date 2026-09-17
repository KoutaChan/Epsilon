import type { ReviewRound } from "../../api/types";

function isProbabilityLower(value: number, reference: number): boolean {
  // 確率の合算で生じる binary64 の丸め差だけを許容し、表示桁数では丸めない。
  return reference - value > Number.EPSILON * Math.max(value, reference);
}

export function summarizePlayerDecisions(
  rounds: readonly Pick<ReviewRound, "steps">[],
): { reviewed: number; matched: number; badMoves: number }[] {
  const players = Array.from({ length: 4 }, () => ({
    reviewed: 0,
    matched: 0,
    badMoves: 0,
  }));
  for (const round of rounds) {
    for (const step of round.steps) {
      if (!step.decision) continue;
      const probabilities = new Map<string | number, number>();
      let chosenKey: string | number | undefined;
      for (const candidate of step.candidates) {
        const key =
          candidate.type === "DAHAI" || candidate.type === "RIICHI_DAHAI"
            ? `${candidate.type}:${candidate.tiles[0]}`
            : candidate.actionId;
        probabilities.set(
          key,
          (probabilities.get(key) ?? 0) + candidate.probability,
        );
        // 同じ打牌のどちらかが実行されていれば、集約した打牌を実際の選択とする。
        if (candidate.chosen) chosenKey = key;
      }
      if (chosenKey === undefined || probabilities.size < 2) continue;
      const chosenProbability = probabilities.get(chosenKey)!;
      let maximumProbability = 0;
      for (const probability of probabilities.values())
        maximumProbability = Math.max(maximumProbability, probability);
      const stats = players[step.actor];
      stats.reviewed++;
      if (!isProbabilityLower(chosenProbability, maximumProbability))
        stats.matched++;
      else if (isProbabilityLower(chosenProbability, 0.05)) stats.badMoves++;
    }
  }
  return players;
}
