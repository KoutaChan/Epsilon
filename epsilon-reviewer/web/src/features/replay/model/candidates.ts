import {
  actionOf,
  candidateLosses,
  candidateProbabilities,
  candidateTiles,
  type CandidateSet,
} from "../../../../../protocol/result.ts";
import type { Candidate } from "./types.ts";

const cache = new WeakMap<CandidateSet, readonly Candidate[]>();

/**
 * 判断の候補の行を表示の候補にする。同じ判断（CandidateSet の物）は初めて見た時に 1 回だけ作る。
 * temperature は文書の温度で、その文書の判断ともしもの答えだけを渡す。
 */
export function readCandidates(
  set: CandidateSet,
  temperature: number | undefined,
): readonly Candidate[] {
  const cached = cache.get(set);
  if (cached) return cached;
  const probabilities = candidateProbabilities(set, temperature);
  const losses = candidateLosses(set, temperature);
  const candidates = set.candidates.map((row, index): Candidate => {
    const { type, tsumogiri } = actionOf(row[0]);
    return {
      actionId: row[0],
      type,
      tiles: candidateTiles(row),
      probability: probabilities[index]!,
      valueLoss: losses?.[index] ?? null,
      chosen: index === set.chosen,
      tsumogiri,
    };
  });
  cache.set(set, candidates);
  return candidates;
}

/** 候補の欄に出す上位の件数。 */
const LISTED = 3;

/**
 * 候補の欄に出す行。上位 3 件と、上位にない実際の選択（本来の順位で末尾に区切って出す）。もしもの比べの間は実際の選択を
 * 出さない（元の局面の手で、比べの行で高くなった欄を画面に収める）。
 */
export function listedCandidates(
  candidates: readonly Candidate[],
  comparing: boolean,
): { top: readonly Candidate[]; outside: Candidate | null } {
  const top = candidates.slice(0, LISTED);
  const outside = comparing
    ? null
    : (candidates.slice(LISTED).find((candidate) => candidate.chosen) ?? null);
  return { top, outside };
}
