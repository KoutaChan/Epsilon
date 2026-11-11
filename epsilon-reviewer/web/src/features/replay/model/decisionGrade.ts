import type { Candidate } from "./types.ts";

// 判断を評価するしきい値は全てここに置き、候補の表示と集計で同じ値を使う。

/** 最善手との価値の差（効用）がこれ未満なら、最善とほぼ同じとみなす。 */
export const NEAR_LOSS = 0.002;
/** 最善手との価値の差（効用）がこれ以上なら悪手とみなす。 */
export const MISTAKE_LOSS = 0.005;
/** 確率だけを出すモデルで、第 1 候補でない手の選択率がこれ未満なら悪手とみなす。 */
export const BAD_PROBABILITY = 0.05;

export type DecisionGrade = "best" | "near" | "inaccuracy" | "mistake";
export type ProbabilityGrade = "top" | "other" | "bad";

/** 価値を推定したモデルの局面か（候補に価値の差があるか）を返す。1 つの局面の候補は、全てが価値の差を持つか、全てが持たないかのどちらか。 */
export function hasValueLoss(candidates: readonly Candidate[]): boolean {
  return candidates.length > 0 && candidates[0]!.valueLoss !== null;
}

export function gradeLoss(loss: number): DecisionGrade {
  if (loss <= 0) return "best";
  if (loss < NEAR_LOSS) return "near";
  if (loss < MISTAKE_LOSS) return "inaccuracy";
  return "mistake";
}

function isProbabilityLower(value: number, reference: number): boolean {
  // 確率を足すと binary64 の丸め誤差が出るので、その程度の差は同じとみなす。表示の桁数では丸めない。
  return reference - value > Number.EPSILON * Math.max(value, reference);
}

/** 選択率が第 1 候補と同じなら一致とする。第 1 候補より低く、しかも悪手のしきい値（{@link BAD_PROBABILITY}）未満なら悪手とする。 */
export function gradeProbability(
  probability: number,
  maximum: number,
): ProbabilityGrade {
  if (!isProbabilityLower(probability, maximum)) return "top";
  return isProbabilityLower(probability, BAD_PROBABILITY) ? "bad" : "other";
}
