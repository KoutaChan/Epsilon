package com.epsilon.grp.train;

import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.Objects;

/** コホート 1 つ分の、主経路の半荘の記録。集めたときに使った Φ の項目の id も持つ。GameRecordCodec の 1 ファイルにあたる。 */
public record GameCohort(Variant variant, int cohort, String phiId, List<GameRecord> games) {

  public GameCohort {
    Objects.requireNonNull(variant, "variant");
    Objects.requireNonNull(phiId, "phiId");
    games = List.copyOf(games);
    for (GameRecord game : games) {
      if (game.variant() != variant) {
        throw new IllegalArgumentException(
            "Cohort " + cohort + " of " + variant + " holds a " + game.variant() + " game");
      }
    }
  }
}
