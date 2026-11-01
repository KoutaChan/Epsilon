import type { Candidate } from "../../api/types";

// 判断の評価基準はすべてここに置き、候補表示と集計で同じ境界を使う。

/** 最善手との価値差（効用）がこれ未満なら最善とほぼ同等とみなす。 */
export const NEAR_LOSS = 0.002;
/** 最善手との価値差（効用）がこれ以上なら悪手とみなす。 */
export const MISTAKE_LOSS = 0.005;
/** 確率だけのモデルで、第1候補と異なりこの選択率未満なら悪手とみなす。 */
export const BAD_PROBABILITY = 0.05;

export type DecisionGrade = "best" | "near" | "inaccuracy" | "mistake";
export type ProbabilityGrade = "top" | "other" | "bad";

/** 価値を推定したモデルの局面か。局面内の候補は全件が値を持つか全件が持たない。 */
export function hasValueLoss(candidates: readonly Candidate[]): boolean {
  return candidates.length > 0 && candidates[0].valueLoss !== null;
}

export function gradeLoss(loss: number): DecisionGrade {
  if (loss <= 0) return "best";
  if (loss < NEAR_LOSS) return "near";
  if (loss < MISTAKE_LOSS) return "inaccuracy";
  return "mistake";
}

function isProbabilityLower(value: number, reference: number): boolean {
  // 確率の合算で生じる binary64 の丸め差だけを許容し、表示桁数では丸めない。
  return reference - value > Number.EPSILON * Math.max(value, reference);
}

/** 同率の第1候補は一致とし、悪手は第1候補と異なる低い選択率に限る。 */
export function gradeProbability(
  probability: number,
  maximum: number,
): ProbabilityGrade {
  if (!isProbabilityLower(probability, maximum)) return "top";
  return isProbabilityLower(probability, BAD_PROBABILITY) ? "bad" : "other";
}
