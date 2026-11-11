package com.epsilon.agent.eval;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.regex.Pattern;

/** 評価の卓で、1 つの牌山を何半荘、どの席の割り当てで打つか。 */
@Choice(key = "kind", doc = "1 つの牌山で打つ半荘と席の割り当て方")
public sealed interface SeatingSettings {

  /** 決闘の席の決め方。 */
  Seating seating();

  /**
   * 決闘の分け方（候補と相手の 2 スロット）。
   *
   * @throws IllegalArgumentException 分け方の席の数の和が卓の席の数でないとき
   */
  List<Split> splits(Variant variant);

  /** 牌山ごとに 1 半荘打つ。候補は牌山の番号を席の数で割った余りの席に 1 人座り、ほかの席は相手。 */
  @Case("BY_GAME")
  record ByGame() implements SeatingSettings {

    @Override
    public Seating seating() {
      return Seating.BY_GAME;
    }

    /** @return 候補が 1 席、相手が残りの席の分け方 1 つだけ */
    @Override
    public List<Split> splits(Variant variant) {
      return List.of(Split.oneAgainstRest(variant));
    }
  }

  /**
   * 牌山ごとに、全ての分け方の全ての席の割り当てを 1 半荘ずつ打つ。
   *
   * @param splits 分け方（スロットごとの席の数を v でつないだもの。{@code 1v2} は候補 1 席と相手 2 席）
   */
  @Case("EVERY_PLACEMENT")
  record EveryPlacement(
      @Key(doc = "分け方（スロットごとの席の数を v でつないだもの。1v2 は候補 1 席と相手 2 席）", presence = Presence.REQUIRED)
          List<String> splits)
      implements SeatingSettings {

    private static final Pattern SPLIT = Pattern.compile("[1-9][0-9]*(v[1-9][0-9]*)+");

    public EveryPlacement {
      splits = List.copyOf(splits);
      if (splits.isEmpty()) {
        throw new InvalidSetting("splits", "at least one split is needed");
      }
      for (String split : splits) {
        if (!SPLIT.matcher(split).matches()) {
          throw new InvalidSetting("splits", "a split is written like 1v2 or 2v1v1, not " + split);
        }
      }
      if (splits.stream().distinct().count() < splits.size()) {
        throw new InvalidSetting("splits", "the splits repeat: " + splits);
      }
    }

    /**
     * 分け方の席の数の和。
     *
     * @param split 分け方（{@code 1v2} の形）
     */
    public static int seats(String split) {
      int seats = 0;
      for (String part : split.split("v")) {
        seats += Integer.parseInt(part);
      }
      return seats;
    }

    @Override
    public Seating seating() {
      return Seating.EVERY_PLACEMENT;
    }

    /** @return 書いた順の分け方 */
    @Override
    public List<Split> splits(Variant variant) {
      return splits.stream().map(split -> Split.parse(split, 2, variant)).toList();
    }
  }
}
