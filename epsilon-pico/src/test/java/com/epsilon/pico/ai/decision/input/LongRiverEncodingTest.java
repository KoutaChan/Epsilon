package com.epsilon.pico.ai.decision.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import java.nio.ShortBuffer;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * ルール上は鳴きで他家の手番が飛ぶと1席の打牌が入力の河の枠（{@link DecisionInputSchema#MAX_RIVER_EVENTS_PER_PLAYER}）を超えうる。そのときも
 * 河のトークンが他席の枠へはみ出さず、すべてのカテゴリ値が特徴ごとの語彙に収まることを確かめる。
 */
public class LongRiverEncodingTest {
  private static final int LONG = DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + 6;

  @Test
  public void longRiverStaysInItsSlotsAndVocabularies() {
    var engine = new GameEngine(901);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    GameState state = engine.getState();
    int observer = point.player();
    int longSeat = (observer + 1) % 4, shortSeat = (observer + 2) % 4;
    // どの牌種も4枚を超えて見えないように、手牌と表示牌にない牌から捨てる。
    int[] left = new int[Tile.NUM_TILE_TYPES];
    Arrays.fill(left, Tile.TILES_PER_TYPE);
    for (int seat = 0; seat < 4; seat++)
      for (int type = 0; type < Tile.NUM_TILE_TYPES; type++) left[type] -= state.hand(seat).count(type);
    left[state.doraState().indicatorTileType(0)]--;
    int[] shortRiver = new int[3];
    int type = 0;
    for (int i = 0; i < LONG + shortRiver.length; i++) {
      while (left[type % Tile.NUM_TILE_TYPES] == 0) type++;
      int discard = type % Tile.NUM_TILE_TYPES;
      left[discard]--;
      type++;
      if (i < LONG) state.commitDiscard(longSeat, discard, false);
      else state.commitDiscard(shortSeat, shortRiver[i - LONG] = discard, false);
    }
    Assert.assertEquals(state.river(longSeat).size(), LONG);

    var builder =
        DecisionBatchBuilder.inference(1, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
    builder.encodeObservedInferenceRow(0, engine.observation(observer), point.legalActions());
    DecisionHostBatch host = builder.buildEncodedInferenceRows();
    var categories = ShortBuffer.allocate(host.sliceRows(0, 1).inputCategoricalElementCount());
    host.sliceRows(0, 1).copyInputCategoriesTo(categories);

    int base =
        DecisionInputSchema.ROUND_INT_COUNT
            + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_INT_STRIDE
            + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_INT_STRIDE;
    int longRelative = 1, shortRelative = 2;
    Assert.assertEquals(
        host.playerCategory(0, longRelative, DecisionInputSchema.PlayerInt.RIVER_COUNT),
        DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER);
    // 長い河は最初の枠数だけ、短い河は自分の3枚だけ。はみ出しがあれば短い河の4枚目以降に PRESENT が立つ。
    for (int slot = 0; slot < DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER; slot++) {
      Assert.assertEquals(river(categories, base, longRelative, slot, DecisionInputSchema.RiverInt.PRESENT), 1);
      Assert.assertEquals(
          river(categories, base, longRelative, slot, DecisionInputSchema.RiverInt.INDEX), slot + 1);
      boolean present = slot < shortRiver.length;
      Assert.assertEquals(
          river(categories, base, shortRelative, slot, DecisionInputSchema.RiverInt.PRESENT),
          present ? 1 : 0,
          "short river slot " + slot);
      if (present)
        Assert.assertEquals(
            river(categories, base, shortRelative, slot, DecisionInputSchema.RiverInt.TILE),
            shortRiver[slot] + 1);
    }
    assertWithinVocabularies(categories);
  }

  private static int river(
      ShortBuffer categories, int base, int relativeSeat, int slot, DecisionInputSchema.RiverInt field) {
    return categories.get(
        base
            + (relativeSeat * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + slot)
                * DecisionInputSchema.RIVER_INT_STRIDE
            + field.ordinal());
  }

  /** 状態のカテゴリ値が、それぞれの特徴の語彙（次の特徴の名前空間オフセットまで）に収まる。 */
  private static void assertWithinVocabularies(ShortBuffer categories) {
    int[] offsets = DecisionCategoryLayout.stateOffsets();
    int[] bounds = Arrays.stream(offsets).distinct().sorted().toArray();
    for (int element = 0; element < offsets.length; element++) {
      int at = Arrays.binarySearch(bounds, offsets[element]);
      int end = at + 1 < bounds.length ? bounds[at + 1] : DecisionCategoryLayout.STATE_DICTIONARY_SIZE;
      int value = categories.get(element);
      Assert.assertTrue(
          value >= 0 && offsets[element] + value < end,
          "state categorical " + element + " = " + value + " exceeds " + (end - offsets[element]));
    }
  }
}
