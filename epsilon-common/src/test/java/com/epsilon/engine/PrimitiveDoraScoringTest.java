package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBatchBuffer;
import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * {@link HandScoreEvaluator} の表ドラを基本型で受ける採点と、裏ドラの数え方。翻・符・基本点・ドラの数を手で数えた値と比べる。基本型の口に
 * 渡す表ドラの列は、局面の外で {@link DoraState#packTile} から組み立てる。
 */
public class PrimitiveDoraScoringTest {

  /** 平和・断么の 13 枚（5p の和了は 345p・567p のどちらでも両面）。 */
  private static final String PINFU_TANYAO = "234567m34567p88s";

  /** 物理の牌山で初めのドラ表示を置く位置（Wall の配列の 130）。 */
  private static final int INDICATOR_POSITION = 130;

  /** 物理の牌山で初めの裏ドラ表示を置く位置（Wall の配列の 131）。 */
  private static final int URA_INDICATOR_POSITION = 131;

  @DataProvider(name = "hands")
  public Object[][] hands() {
    return new Object[][] {
      // 13 枚の手、白のポンの有無、和了牌、ドラ表示牌、表ドラの数、
      // 条件ごと（立直ロン・立直ツモ・ダマロン・ダマツモ）の {翻, 符, 基本点}
      {
        PINFU_TANYAO,
        false,
        Tile.P5,
        new int[] {Tile.P4, Tile.M1},
        3,
        new int[][] {{6, 30, 3000}, {7, 20, 3000}, {5, 30, 2000}, {6, 20, 3000}}
      },
      {
        PINFU_TANYAO,
        false,
        Tile.P5,
        new int[] {Tile.M1, Tile.M1, Tile.P4, Tile.S7, Tile.HAKU},
        6,
        new int[][] {{9, 30, 4000}, {10, 20, 4000}, {8, 30, 4000}, {9, 20, 4000}}
      },
      {
        "234m567p23s88s",
        true,
        Tile.S4,
        new int[] {Tile.S3, Tile.S3},
        2,
        new int[][] {{3, 30, 960}, {3, 30, 960}, {3, 30, 960}, {3, 30, 960}}
      },
      {
        "1122m3344p5566s7z",
        false,
        Tile.CHUN,
        new int[] {Tile.HATSU},
        2,
        new int[][] {{5, 25, 2000}, {6, 25, 3000}, {4, 25, 1600}, {5, 25, 2000}}
      }
    };
  }

  /** DoraState の口と、表示牌から組み立てた表ドラの列を基本型で渡す口が、どちらも手で数えた値になる。 */
  @Test(dataProvider = "hands")
  public void primitiveDoraScoresLikeHandCount(
      String concealed,
      boolean ponHaku,
      int winningTile,
      int[] indicators,
      int expectedDora,
      int[][] expected) {
    GameState state = MahjongFixtures.state();
    state.initializeWallForReconstruction(indicators, 0);
    Hand hand = hand(concealed, ponHaku);
    long packed = 0L;
    for (int index = 0; index < indicators.length; index++) {
      packed = DoraState.packTile(packed, index, Tile.doraFrom(indicators[index]));
    }

    HandScoreEvaluator evaluator = new HandScoreEvaluator();
    HandScoreBatchBuffer viaState = conditions();
    HandScoreBatchBuffer viaPrimitive = conditions();
    evaluator.scoreAfterAddingBatch(hand, hand, winningTile, state.doraState(), false, viaState);
    evaluator.scoreAfterAddingBatch(
        hand, hand, winningTile, packed, indicators.length, viaPrimitive);
    Assert.assertEquals(packed, state.doraState().doraTileTypesPacked());
    for (int index = 0; index < expected.length; index++) {
      assertScore(viaState.result(index), expected[index], expectedDora, 0);
      assertScore(viaPrimitive.result(index), expected[index], expectedDora, 0);
    }
  }

  /** 立直で裏が分かっている局面（表示 1m・裏表示 7s、手に 2m が 1 枚と 8s が 2 枚）。裏ドラは立直の和了で、求めたときだけ数える。 */
  @Test
  public void countsUraOnlyWhenRequestedInRiichi() {
    GameState state = MahjongFixtures.state();
    state.initializeWall(wallWithIndicators(Tile.M1, Tile.S7));
    DoraState dora = state.doraState();
    Hand hand = hand(PINFU_TANYAO, false);
    // 条件ごと（立直ロン・立直ツモ・ダマロン・ダマツモ）の {翻, 符, 基本点} と裏ドラの数
    int[][] withUra = {{6, 30, 3000}, {7, 20, 3000}, {3, 30, 960}, {4, 20, 1280}};
    int[] ura = {2, 2, 0, 0};
    int[][] withoutUra = {{4, 30, 1920}, {5, 20, 2000}, {3, 30, 960}, {4, 20, 1280}};

    HandScoreEvaluator evaluator = new HandScoreEvaluator();
    HandScoreBatchBuffer batch = conditions();
    evaluator.scoreAfterAddingBatch(hand, hand, Tile.P5, dora, true, batch);
    for (int index = 0; index < withUra.length; index++) {
      assertScore(batch.result(index), withUra[index], 1, ura[index]);
    }
    evaluator.scoreAfterAddingBatch(hand, hand, Tile.P5, dora, false, batch);
    for (int index = 0; index < withoutUra.length; index++) {
      assertScore(batch.result(index), withoutUra[index], 1, 0);
    }

    Hand completed = hand(PINFU_TANYAO, false);
    completed.add(Tile.P5);
    WinConditions riichi = WinConditions.publicDecision(Tile.NAN, Tile.TON, RiichiState.RIICHI);
    HandScoreBuffer single = new HandScoreBuffer();
    evaluator.scoreCompleted(completed, Tile.P5, WinMethod.RON, riichi, dora, true, 0, single);
    assertScore(single, withUra[0], 1, ura[0]);
    evaluator.scoreCompleted(completed, Tile.P5, WinMethod.RON, riichi, dora, false, 0, single);
    assertScore(single, withoutUra[0], 1, 0);
  }

  @Test
  public void packTileIsTheInverseOfPackedTile() {
    long packed = DoraState.packTile(DoraState.packTile(0L, 0, Tile.CHUN), 4, Tile.S9);
    Assert.assertEquals(DoraState.packedTile(packed, 0), Tile.CHUN);
    Assert.assertEquals(DoraState.packedTile(packed, 4), Tile.S9);
    long replaced = DoraState.packTile(packed, 0, Tile.P5);
    Assert.assertEquals(DoraState.packedTile(replaced, 0), Tile.P5);
    Assert.assertEquals(DoraState.packedTile(replaced, 4), Tile.S9);
    Assert.expectThrows(
        IndexOutOfBoundsException.class,
        () -> DoraState.packTile(0L, DoraState.MAX_INDICATORS, Tile.M1));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> DoraState.packTile(0L, 0, Tile.NUM_TILE_TYPES));
  }

  @Test
  public void rejectsDoraOutsideTheRules() {
    HandScoreEvaluator evaluator = new HandScoreEvaluator();
    Hand hand = hand(PINFU_TANYAO, false);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> evaluator.scoreAfterAddingBatch(hand, hand, Tile.P5, 0L, 6, conditions()));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> evaluator.scoreAfterAddingBatch(hand, hand, Tile.P5, 63L, 1, conditions()));
  }

  private static void assertScore(
      HandScoreBuffer score, int[] hanFuBase, int omoteDora, int uraDora) {
    Assert.assertTrue(score.available());
    Assert.assertEquals(score.han(), hanFuBase[0], "han");
    Assert.assertEquals(score.fu(), hanFuBase[1], "fu");
    Assert.assertEquals(score.basePoints(), hanFuBase[2], "base points");
    Assert.assertEquals(score.omoteDoraCount(), omoteDora, "omote dora");
    Assert.assertEquals(score.uraDoraCount(), uraDora, "ura dora");
  }

  private static Hand hand(String concealed, boolean ponHaku) {
    GameState state = MahjongFixtures.state();
    MahjongFixtures.hand(state, 1, concealed);
    if (ponHaku) {
      MahjongFixtures.call(state, 1, Meld.pon(Tile.HAKU, Meld.RelativeSource.SHIMOCHA));
    }
    return state.hand(1);
  }

  /** 初めのドラ表示と裏表示に指定の牌（赤でない写し）を置いた、136 枚の物理牌の並び。 */
  private static int[] wallWithIndicators(int indicator, int uraIndicator) {
    int[] ids = new int[Tile.NUM_TILES];
    for (int id = 0; id < ids.length; id++) ids[id] = id;
    swap(ids, INDICATOR_POSITION, indicator * Tile.TILES_PER_TYPE + 1);
    swap(ids, URA_INDICATOR_POSITION, uraIndicator * Tile.TILES_PER_TYPE + 1);
    return ids;
  }

  private static void swap(int[] ids, int first, int second) {
    int kept = ids[first];
    ids[first] = ids[second];
    ids[second] = kept;
  }

  /** 南家・東場のロンとツモを、立直とダマで。 */
  private static HandScoreBatchBuffer conditions() {
    WinConditions riichi = WinConditions.publicDecision(Tile.NAN, Tile.TON, RiichiState.RIICHI);
    WinConditions dama = WinConditions.publicDecision(Tile.NAN, Tile.TON, RiichiState.NONE);
    return new HandScoreBatchBuffer(4)
        .size(4)
        .configure(0, WinMethod.RON, riichi, 0)
        .configure(1, WinMethod.TSUMO, riichi, 0)
        .configure(2, WinMethod.RON, dama, 0)
        .configure(3, WinMethod.TSUMO, dama, 0);
  }
}
