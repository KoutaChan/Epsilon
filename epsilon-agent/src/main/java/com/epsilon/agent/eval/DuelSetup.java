package com.epsilon.agent.eval;

import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 決闘のジョブの計画（job.json の plan）。ジョブを初めて始めるときに 1 回だけ決めて書き込み、続きから打つときはこれを読む。打ち手の
 * チェックポイント、牌山、席の割り当て、判定の時点と臨界値をここで固定するので、何度止めて続けても、同じ牌山を同じ境界で判定する。
 *
 * @param candidate スロット 0 の打ち手
 * @param opponent スロット 1 の打ち手
 * @param splits 席の分け方（"1v3" の形。割り当ては分け方の順に数える）
 * @param draws ツモの順
 * @param wallSeed 牌山の種の素（卓は {@link Arena#EVALUATION_SALT} を混ぜて使う）
 * @param wallBase 最初の牌山の番号
 * @param walls 牌山の数。止め方が精度か判定なら上限
 * @param chunk チャンクの牌山の数（{@link DuelPlan#chunk}）
 * @param looks 判定の時点（早い順）
 */
public record DuelSetup(
    Variant variant,
    Entrant candidate,
    Entrant opponent,
    Seating seating,
    List<String> splits,
    DrawOrder draws,
    WallSet wallSet,
    long wallSeed,
    long wallBase,
    int walls,
    StopRule stop,
    Score score,
    RankUtility utility,
    int chunk,
    List<Look> looks) {

  /**
   * 決闘の打ち手。系譜のバージョンは書き換えないので、指定とバージョンで見分ける。系譜の外のチェックポイントは、計画したときの中身の SHA-256
   * も持ち、開くときに照合する。
   *
   * @param spec 打ち手の指定（run の種類ごとに、これを見て方策を開く）
   * @param label 結果に書く名前
   * @param version 打ち手の系譜のバージョン。系譜の外の打ち手には無い
   * @param sha256 系譜の外のチェックポイントの SHA-256。系譜のバージョンと、チェックポイントを開かない打ち手には無い
   */
  public record Entrant(
      String spec,
      String label,
      @Key(doc = "打ち手の系譜のバージョン", presence = Presence.ABSENT) Integer version,
      @Key(doc = "系譜の外のチェックポイントの SHA-256", presence = Presence.ABSENT) String sha256) {

    public Entrant {
      if (spec == null || spec.isBlank() || label == null || label.isBlank()) {
        throw new IllegalArgumentException("A duel entrant needs a spec and a label: " + spec + ", " + label);
      }
      if (version != null && sha256 != null) {
        throw new IllegalArgumentException(
            "Version " + version + " of the lineage is told apart by its version, not by a SHA-256: " + label);
      }
    }
  }

  /**
   * 判定の時点。
   *
   * @param walls その時点までに打ち終える牌山の数
   * @param critical 区間の臨界値
   */
  public record Look(int walls, double critical) {}

  public DuelSetup {
    Objects.requireNonNull(candidate, "candidate");
    Objects.requireNonNull(opponent, "opponent");
    Objects.requireNonNull(draws, "draws");
    Objects.requireNonNull(wallSet, "wallSet");
    Objects.requireNonNull(score, "score");
    Objects.requireNonNull(utility, "utility");
    splits = List.copyOf(splits);
    looks = List.copyOf(looks);
    DuelPlan.check(variant, 2, seating, walls, stop);
    if (splits.isEmpty()) {
      throw new IllegalArgumentException("A duel needs a split");
    }
    for (String split : splits) {
      Split.parse(split, 2, variant);
    }
    if (chunk != DuelPlan.chunk(stop, walls)) {
      throw new IllegalArgumentException("A duel of " + walls + " walls plays chunks of " + DuelPlan.chunk(stop, walls)
          + " walls, not " + chunk);
    }
    Stopping stopping = Stopping.recorded(stop, walls, criticals(looks));
    for (int look = 0; look < looks.size(); look++) {
      if (looks.get(look).walls() != stopping.wallsAt(look)) {
        throw new IllegalArgumentException(
            "Look " + look + " comes after " + stopping.wallsAt(look) + " walls, not " + looks.get(look).walls());
      }
    }
  }

  /**
   * 判定の時点と臨界値を求めて、計画を作る。
   *
   * @param splits 席の分け方（割り当てを数える順）
   * @return 計画
   * @throws IllegalArgumentException 決闘の形が正しくないとき（{@link DuelPlan}）
   */
  public static DuelSetup of(
      Variant variant,
      Entrant candidate,
      Entrant opponent,
      Seating seating,
      List<Split> splits,
      DrawOrder draws,
      WallSet wallSet,
      long wallSeed,
      long wallBase,
      int walls,
      StopRule stop,
      Score score,
      RankUtility utility) {
    DuelPlan.check(variant, 2, seating, walls, stop);
    Stopping stopping = Stopping.of(stop, walls);
    List<Look> looks = new ArrayList<>(stopping.looks());
    for (int look = 0; look < stopping.looks(); look++) {
      looks.add(new Look(stopping.wallsAt(look), stopping.critical(look)));
    }
    return new DuelSetup(
        variant,
        candidate,
        opponent,
        seating,
        splits.stream().map(Split::toString).toList(),
        draws,
        wallSet,
        wallSeed,
        wallBase,
        walls,
        stop,
        score,
        utility,
        DuelPlan.chunk(stop, walls),
        looks);
  }

  /**
   * @param candidate スロット 0 の方策。四人打ちか三人打ちかは、この計画と同じであること
   * @param opponent スロット 1 の方策
   * @return 方策を置いた決闘の計画
   */
  public DuelPlan plan(Policy candidate, Policy opponent) {
    return new DuelPlan(
        new ArenaPlan(
            variant,
            List.of(candidate, opponent),
            seating,
            layout().splits(),
            draws,
            wallSet,
            wallSeed,
            wallBase,
            walls,
            utility),
        stop,
        score);
  }

  /** @return 記録した分け方での席の割り当て。打ち手を開かずに作れる */
  public Placements layout() {
    List<Split> parsed = new ArrayList<>(splits.size());
    for (String split : splits) {
      parsed.add(Split.parse(split, 2, variant));
    }
    return new Placements(variant, seating, parsed);
  }

  /**
   * 2 つの計画で、牌山ごとの値を同じ牌山どうしで対にできるかを返す。同じ相手（系譜のバージョンなら同じ指定とバージョン）と、同じ牌山・
   * 席の割り当て・ツモの順で打ち、同じ効用で値を出すときだけ対にできる。候補・牌山の数・止め方は違ってよい（対にするのは両方で打った牌山
   * だけ）。
   *
   * @param other もう 1 つの計画
   * @return 対にできるなら true
   */
  public boolean pairsWith(DuelSetup other) {
    return variant == other.variant
        && opponent.equals(other.opponent)
        && seating == other.seating
        && splits.equals(other.splits)
        && draws == other.draws
        && wallSet == other.wallSet
        && wallSeed == other.wallSeed
        && wallBase == other.wallBase
        && utility == other.utility;
  }

  /** @return 計画に書いた判定の時点と臨界値を使う止め方 */
  public Stopping stopping() {
    return Stopping.recorded(stop, walls, criticals(looks));
  }

  private static double[] criticals(List<Look> looks) {
    double[] critical = new double[looks.size()];
    for (int look = 0; look < critical.length; look++) {
      critical[look] = looks.get(look).critical();
    }
    return critical;
  }
}
