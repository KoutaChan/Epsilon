package com.epsilon.agent.eval;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * 決闘の結果（result.json と報告の行）。値は止めた時点のもの。判定で止めたとき（{@link Stopped#DECIDED}）は、早く止まった決闘ほど
 * 平均が真の差より大きめに出るので、強さの指標としては読まない。
 *
 * @param walls 使った牌山の数
 * @param games 使った半荘の数
 * @param verdict 判定（候補が良い向き）
 * @param stopped 止めた理由
 * @param score 判定に使った量
 * @param critical 止めた時点の区間の臨界値
 * @param rank 順位の対の量 δ（牌山ごとに候補と相手を比べた値。負が良い）の平均・標準誤差・区間
 * @param utility 効用の対の量 υ（牌山ごとに候補と相手を比べた値。正が良い）の平均・標準誤差・区間
 * @param splits 席の分け方ごとの値（計画の分け方の順）
 * @param slots スロットごとの平均（スロットの順。0 が候補、1 が相手の 2 つ）
 * @param play 打ち方の差（候補の側 − 相手の側。1 席・1 局あたりの率）
 * @param identical 全ての割り当ての半荘が同じ点で終わった牌山の割合（牌山 1 つで 1 半荘だけ打つ卓では 0）
 * @param roundSpread 牌山の中で割り当てごとの局の数がどれだけばらつくか。牌山ごとの標準偏差を、全ての牌山で平均した値（牌山 1 つで
 *     1 半荘だけ打つ卓では 0）
 * @param seconds この試行で卓を打つのにかかった秒数
 */
public record DuelResult(
    int walls,
    long games,
    Verdict verdict,
    Stopped stopped,
    Score score,
    double critical,
    Estimate rank,
    Estimate utility,
    List<SplitValue> splits,
    List<SlotValue> slots,
    PlayStats.Rates play,
    double identical,
    double roundSpread,
    double seconds) {

  /**
   * 席の分け方 1 つの値。どちらも候補 1 席・相手（席の数 − 1）席の卓に直した値で、区間の臨界値は決闘の結果全体と同じ。
   *
   * @param split 席の分け方（"1v3" の形）
   * @param rank δ
   * @param utility υ
   */
  public record SplitValue(String split, Estimate rank, Estimate utility) {}

  /**
   * スロット 1 つの平均。値はスロットの席の 1 席・1 半荘あたりで、誤差は牌山ごとの値から求めた標準誤差。
   *
   * @param label 打ち手の名前
   * @param meanRank 平均の着順（1 始まり）
   * @param meanUtility 平均の効用
   * @param shares 着順ごとの割合（席の数だけ）
   */
  public record SlotValue(
      String label, double meanRank, double rankError, double meanUtility, double utilityError, List<Double> shares) {

    public SlotValue {
      shares = List.copyOf(shares);
    }
  }

  public DuelResult {
    Objects.requireNonNull(verdict, "verdict");
    Objects.requireNonNull(stopped, "stopped");
    Objects.requireNonNull(score, "score");
    Objects.requireNonNull(rank, "rank");
    Objects.requireNonNull(utility, "utility");
    Objects.requireNonNull(play, "play");
    splits = List.copyOf(splits);
    slots = List.copyOf(slots);
    if (slots.size() != 2) {
      throw new IllegalArgumentException(
          "A duel result holds the slots of the candidate and the opponent, not " + slots.size() + " slot(s)");
    }
  }

  /** @return 人が読むための 1 行（スクリプトはこれを読まない） */
  public String summary() {
    SlotValue candidate = slots.get(0);
    return String.format(
        Locale.ROOT,
        "candidate=%s opponent=%s walls=%d games=%d rank=%+.4f±%.4f [%+.4f,%+.4f] utility=%+.4f±%.4f verdict=%s"
            + " stopped=%s meanRank=%.4f±%.4f win=%+.4f dealIn=%+.4f riichi=%+.4f open=%+.4f nuki=%+.4f"
            + " identical=%.3f roundSpread=%.3f seconds=%.1f",
        candidate.label(),
        slots.get(1).label(),
        walls,
        games,
        rank.mean(),
        rank.error(),
        rank.low(),
        rank.high(),
        utility.mean(),
        utility.error(),
        verdict,
        stopped,
        candidate.meanRank(),
        candidate.rankError(),
        play.win(),
        play.dealIn(),
        play.riichi(),
        play.open(),
        play.nuki(),
        identical,
        roundSpread,
        seconds);
  }
}
