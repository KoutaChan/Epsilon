package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.RankPermutations;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 主経路の半荘 1 つの記録。終局しなかった局末ごとに、次局の卓を局末の順に持つ。最終の点（絶対席。残った供託をトップに渡した後）も
 * 持つ。終局した局末は、最終の点だけで表す。卓と点はコピーして持ち、書き換えない。
 *
 * <p>局末 k で終わった局は、k = 0 なら東 1 局、それ以外は局末 k − 1 の次の局。
 */
public record GameRecord(Variant variant, long gameId, List<GrpState> handEnds, int[] finalScores) {

  public GameRecord {
    Objects.requireNonNull(variant, "variant");
    handEnds = List.copyOf(handEnds);
    for (GrpState state : handEnds) {
      if (state.variant() != variant) {
        throw new IllegalArgumentException("Game " + gameId + " mixes " + state.variant() + " into " + variant);
      }
    }
    if (finalScores.length != variant.seats()) {
      throw new IllegalArgumentException(
          variant + " needs " + variant.seats() + " final scores, got " + finalScores.length);
    }
    finalScores = finalScores.clone();
  }

  public int finalScore(int seat) {
    return finalScores[seat];
  }

  /** 最終順位（0 始まり。同点は起家に近い席が上）。 */
  public int finalRank(int seat) {
    return Placement.rank(finalScores, seat);
  }

  /** その局末で終わった局の番号（局末 0 なら東 1 局）。 */
  public int endedRound(int handEnd) {
    Objects.checkIndex(handEnd, handEnds.size());
    return handEnd == 0 ? 0 : handEnds.get(handEnd - 1).round();
  }

  /** その局末から見た、実際の最終順位の順列の番号。席は次局の親を 0 とする相対席で、番号は RankPermutations に従う。 */
  public int realizedPermutation(int handEnd) {
    return RankPermutations.of(variant).realized(finalScores, handEnds.get(handEnd).dealer());
  }

  /** コピーを返す。 */
  @Override
  public int[] finalScores() {
    return finalScores.clone();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof GameRecord game
        && variant == game.variant
        && gameId == game.gameId
        && handEnds.equals(game.handEnds)
        && Arrays.equals(finalScores, game.finalScores);
  }

  @Override
  public int hashCode() {
    return Objects.hash(variant, gameId, handEnds, Arrays.hashCode(finalScores));
  }

  @Override
  public String toString() {
    return "GameRecord[" + variant + " id=" + gameId + " handEnds=" + handEnds.size()
        + " final=" + Arrays.toString(finalScores) + "]";
  }
}
