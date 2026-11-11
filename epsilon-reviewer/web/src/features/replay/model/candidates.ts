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
 * 判断の候補の行を、表示用の候補に直す。同じ判断（同じ CandidateSet のオブジェクト）の候補は、初めて見たときに 1 回だけ作って覚える。
 * 温度には解析結果の文書の温度を渡す。覚えた候補は温度で区別しないので、渡す判断はその文書の判断ともしもの答えに限る。
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

/** 候補欄に出す上位の件数。 */
const LISTED = 3;

/**
 * 候補欄に出す行を返す。上位 3 件と、上位に入らなかった実際の選択（本来の順位のまま、区切って末尾に出す）。
 * もしもの比較の間は、実際の選択を出さない。実際の選択は元の局面での手で、もしもの局面の手ではないため。出さない分、比較の行で高くなった欄も画面に収まる。
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
