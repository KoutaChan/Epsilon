package com.epsilon.reviewer.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordEvent.AddedKan;
import com.epsilon.mahjong.record.RecordEvent.Call;
import com.epsilon.mahjong.record.RecordEvent.Discard;
import com.epsilon.mahjong.record.RecordEvent.Draw;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.TestRecords;
import java.util.List;

/** 手で作った東 1 局だけの牌譜（親は席 0、四人打ち）。読み込むときに再生で確かめないので、局が途中で終わっていてもよい。 */
final class TestRounds {
  static final Variant VARIANT = Variant.TENHOU_4P;

  private static final int[] SCORES = {25000, 25000, 25000, 25000};

  private TestRounds() {}

  /** 局の結果は、点の動かない流局にする。判断までしか進めないテストに使う。 */
  static LoadedRecord round(int[][] deal, int indicator, RecordEvent... events) {
    return round(deal, indicator, RoundResult.drawWithoutDetails(new int[4]), events);
  }

  static LoadedRecord round(int[][] deal, int indicator, RoundResult result, RecordEvent... events) {
    var start = new RoundStart(VARIANT, 0, 0, 0, SCORES);
    var round = new RecordedRound(start, deal, indicator, List.of(events), result);
    var meta =
        new RecordedGame.Meta(
            "synthetic", RecordedGame.Source.MJAI, List.of("A", "B", "C", "D"));
    return TestRecords.unchecked(new RecordedGame(VARIANT, List.of(round), SCORES, meta));
  }

  /**
   * 天鳳の牌譜には無い流れを通す局。席 1 が席 0 の 5p をポンし、ツモった赤の 5p で加槓する。席 2 は 34p の 25p 待ちで、最初の 5p では
   * 役が無くロンできない。加槓の 5p を席 2 が槍槓でロンし（槍槓・赤 1 の 2 翻 40 符）、局が終わる。
   */
  static LoadedRecord chankanRound() {
    int[][] deal = {
      {55, 17, 18, 32, 33, 36, 37, 84, 85, 108, 109, 112, 116},
      {53, 54, 72, 76, 80, 92, 96, 100, 60, 64, 68, 120, 121},
      {0, 1, 2, 4, 8, 12, 20, 24, 28, 104, 105, 44, 48},
      {9, 13, 19, 21, 25, 29, 41, 56, 77, 81, 86, 124, 132}
    };
    int pon = Meld.pon(Tiles.P5, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE).id();
    var ron =
        new RoundResult(
            EndKind.RON,
            null,
            List.of(RoundResult.Win.unscored(2)),
            1,
            0,
            0,
            new int[] {0, -2600, 2600, 0},
            new int[0]);
    return round(
        deal,
        128,
        ron,
        new Draw(0, 129),
        new Discard(0, 55, false, false),
        new Call(1, pon),
        new Discard(1, 120, false, false),
        new Draw(2, 49),
        new Discard(2, 49, true, false),
        new Draw(3, 130),
        new Discard(3, 130, true, false),
        new Draw(0, 131),
        new Discard(0, 131, true, false),
        new Draw(1, 52),
        new AddedKan(1, Tiles.P5));
  }
}
