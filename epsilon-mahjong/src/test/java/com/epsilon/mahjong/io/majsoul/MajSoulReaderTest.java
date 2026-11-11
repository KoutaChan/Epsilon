package com.epsilon.mahjong.io.majsoul;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.Tiles;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 雀魂の牌譜を読むテスト。手で書いた 1 局だけの半荘（南 4 局、親は席 3）を使う。席 3 の 3p を対面の席 1 がポンし（席 0 もこの 3p を
 * チーできるが、ポンが優先される）、席 1 の 6s に席 2 が断么九・平和でロンして終局する。フィールド名は snake_case と lowerCamelCase を
 * 混ぜてある。打ち手の名前があるのは席 0 と席 2 だけ。包の席は、手で書いた大三元の局で確かめる。
 */
public class MajSoulReaderTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /** 手で書いた半荘の JSON（{@code RecordFiles} のテストも読む）。 */
  public static final String RECORD =
      """
      {"head": {"accounts": [{"seat": 2, "nickname": "West"}, {"seat": 0, "nickname": "East"}, {"seat": 1}],
        "result": {"players": [
        {"seat": 0, "part_point_1": 24000}, {"seat": 1, "partPoint1": 21000},
        {"seat": 2, "part_point_1": 31000}, {"seat": 3, "part_point_1": 24000}]}},
       "records": [
        {"name": ".lq.RecordNewRound", "data": {"chang": 1, "ju": 3, "ben": 0, "liqibang": 0,
          "scores": [24000, 23000, 29000, 24000], "doras": ["9s"],
          "tiles0": ["2p", "4p", "2m", "8m", "2s", "8s", "4z", "4z", "5z", "5z", "6z", "7z", "7z"],
          "tiles1": ["3p", "3p", "3m", "3m", "7m", "7m", "3s", "3s", "6s", "7s", "7s", "6z", "6z"],
          "tiles2": ["2m", "3m", "4m", "5m", "6m", "7m", "3p", "4p", "5p", "6s", "6s", "7s", "8s"],
          "tiles3": ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "1z", "2z", "2z", "3z", "3p", "7z", "8p"]}},
        {"name": ".lq.RecordDiscardTile", "data": {"seat": 3, "tile": "3p", "moqie": false}},
        {"name": ".lq.RecordChiPengGang", "data": {"seat": 1, "type": 1, "tiles": ["3p", "3p", "3p"],
          "froms": [1, 1, 3]}},
        {"name": ".lq.RecordDiscardTile", "data": {"seat": 1, "tile": "6s", "moqie": false}},
        {"name": ".lq.RecordHule", "data": {"hules": [{"seat": 2, "huTile": "6s", "zimo": false,
          "yiman": false, "count": 2, "fu": 30, "fans": [{"id": 12, "val": 1}, {"id": 14, "val": 1}],
          "point_rong": 2000}], "deltaScores": [0, -2000, 2000, 0]}}]}
      """;

  /**
   * 手で書いた大三元の 1 局だけの半荘（東 1 局、親は席 0）の、和了の前まで。席 0 が白と發を暗槓し、席 1 がツモ切った中をポンして、
   * 1p・4p 待ちになる。{@code %s} に、続きの記録を入れる。席 1 が 48000 点を払って飛ぶ。
   */
  private static final String DAISANGEN =
      """
      {"head": {"accounts": [], "result": {"players": [
        {"seat": 0, "part_point_1": 73000}, {"seat": 1, "part_point_1": -23000},
        {"seat": 2, "part_point_1": 25000}, {"seat": 3, "part_point_1": 25000}]}},
       "records": [
        {"name": ".lq.RecordNewRound", "data": {"chang": 0, "ju": 0, "ben": 0, "liqibang": 0,
          "scores": [25000, 25000, 25000, 25000], "doras": ["9p"],
          "tiles0": ["5z", "5z", "5z", "5z", "6z", "6z", "6z", "6z", "7z", "7z", "1m", "2p", "3p", "9s"],
          "tiles1": ["1p", "4p", "7p", "2s", "5s", "8s", "3s", "6s", "9s", "1z", "2z", "3z", "4z"],
          "tiles2": ["2p", "5p", "8p", "1s", "4s", "7s", "3m", "6m", "9m", "1z", "2z", "3z", "4z"],
          "tiles3": ["3p", "6p", "9p", "2s", "5s", "8s", "1m", "4m", "7m", "1z", "2z", "3z", "4z"]}},
        {"name": ".lq.RecordAnGangAddGang", "data": {"seat": 0, "type": 3, "tiles": "5z", "doras": ["9p", "9m"]}},
        {"name": ".lq.RecordDealTile", "data": {"seat": 0, "tile": "9s"}},
        {"name": ".lq.RecordAnGangAddGang", "data": {"seat": 0, "type": 3, "tiles": "6z",
          "doras": ["9p", "9m", "7s"]}},
        {"name": ".lq.RecordDealTile", "data": {"seat": 0, "tile": "8m"}},
        {"name": ".lq.RecordDiscardTile", "data": {"seat": 0, "tile": "1m", "moqie": false}},
        {"name": ".lq.RecordDealTile", "data": {"seat": 1, "tile": "7z"}},
        {"name": ".lq.RecordDiscardTile", "data": {"seat": 1, "tile": "7z", "moqie": true}},
        {"name": ".lq.RecordChiPengGang", "data": {"seat": 0, "type": 1, "tiles": ["7z", "7z", "7z"],
          "froms": [0, 0, 1]}},
        {"name": ".lq.RecordDiscardTile", "data": {"seat": 0, "tile": "8m", "moqie": false}},
        {"name": ".lq.RecordDealTile", "data": {"seat": 1, "tile": "5m"}},
        %s]}
      """;

  /** 大三元の局の続き。席 1・席 2・席 3 がツモ切り、席 0 が 1p をツモる。 */
  private static final String DAISANGEN_TSUMO =
      """
      {"name": ".lq.RecordDiscardTile", "data": {"seat": 1, "tile": "5m", "moqie": true}},
      {"name": ".lq.RecordDealTile", "data": {"seat": 2, "tile": "5m"}},
      {"name": ".lq.RecordDiscardTile", "data": {"seat": 2, "tile": "5m", "moqie": true}},
      {"name": ".lq.RecordDealTile", "data": {"seat": 3, "tile": "5s"}},
      {"name": ".lq.RecordDiscardTile", "data": {"seat": 3, "tile": "5s", "moqie": true}},
      {"name": ".lq.RecordDealTile", "data": {"seat": 0, "tile": "1p"}},
      {"name": ".lq.RecordHule", "data": {"hules": [{"seat": 0, "zimo": true, "yiman": true, "count": 1,
        "fans": [{"id": 37, "val": 1}]}], "delta_scores": [48000, -48000, 0, 0]}}
      """;

  /** 大三元の局の続き。中を鳴かせた席 1 が 1p を打ち、席 0 がロンする。 */
  private static final String DAISANGEN_RON =
      """
      {"name": ".lq.RecordDiscardTile", "data": {"seat": 1, "tile": "1p", "moqie": false}},
      {"name": ".lq.RecordHule", "data": {"hules": [{"seat": 0, "zimo": false, "yiman": true, "count": 1,
        "fans": [{"id": 37, "val": 1}]}], "delta_scores": [48000, -48000, 0, 0]}}
      """;

  /**
   * 大三元の和了の包の席は、3 つ目の三元牌を鳴かせた席。その席が放銃したロンは、包の無いロンになる。どちらの半荘も、記録どおりに
   * 再生できる。
   */
  @Test
  public void theSeatThatFedTheThirdDragonIsLiable() {
    RecordedGame tsumo = MajSoulReader.read(DAISANGEN.formatted(DAISANGEN_TSUMO), "pao-tsumo");
    RecordedGame ron = MajSoulReader.read(DAISANGEN.formatted(DAISANGEN_RON), "pao-feeder-ron");
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    driver.replay(ron, ws);
    driver.replay(tsumo, ws);
    Assert.assertEquals(tsumo.rounds().get(0).result().wins().get(0).pao(), 1);
    Assert.assertEquals(ron.rounds().get(0).result().wins().get(0).pao(), RoundOutcome.NO_SEAT);
  }

  @Test
  public void aHandWrittenRoundReadsAndReplays() {
    RecordedGame game = MajSoulReader.read(RECORD, "pon-over-chi");
    Assert.assertEquals(game.variant(), VARIANT);
    Assert.assertEquals(
        game.meta(),
        new RecordedGame.Meta("pon-over-chi", RecordedGame.Source.MAJSOUL, List.of("East", "", "West", "")));
    Assert.assertEquals(game.finalScores(), new int[] {24_000, 21_000, 31_000, 24_000});
    Assert.assertEquals(game.rounds().size(), 1);
    RecordedRound round = game.rounds().get(0);
    Assert.assertEquals(round.start(), new RoundStart(VARIANT, 7, 0, 0, new int[] {24_000, 23_000, 29_000, 24_000}));
    Assert.assertEquals(round.indicator(), Tiles.physicalId(Tiles.S9, 1));
    Assert.assertEquals(round.deal(3, 11), Tiles.physicalId(Tiles.P3, 3));
    Meld pon = Meld.pon(Tiles.P3, Meld.RelativeSource.TOIMEN, Meld.AkaSource.NONE);
    Assert.assertEquals(
        round.events(),
        List.of(
            new RecordEvent.Draw(3, Tiles.physicalId(Tiles.P8, 0)),
            new RecordEvent.Discard(3, Tiles.physicalId(Tiles.P3, 3), false, false),
            new RecordEvent.Call(1, pon.id()),
            new RecordEvent.Discard(1, Tiles.physicalId(Tiles.S6, 0), false, false)));
    RoundResult.Win win =
        new RoundResult.Win(
            2, 480, ScoringYaku.TANYAO.bit() | ScoringYaku.PINFU.bit(), 2, 30, 0, 0, 0, RoundOutcome.NO_SEAT);
    Assert.assertEquals(
        round.result(),
        new RoundResult(EndKind.RON, null, List.of(win), 1, 0, 0, new int[] {0, -2_000, 2_000, 0}, new int[0]));
    new ReplayDriver(VARIANT).replay(game, new RuleWorkspace(VARIANT));
  }

  /** 最後の打牌と違う牌を鳴いた記録は読まない。 */
  @Test
  public void aCallOfAnotherTileIsRejected() {
    String wrong = RECORD.replace("\"tiles\": [\"3p\", \"3p\", \"3p\"]", "\"tiles\": [\"3p\", \"3p\", \"4p\"]");
    Assert.assertNotEquals(wrong, RECORD);
    Assert.assertThrows(IllegalArgumentException.class, () -> MajSoulReader.read(wrong, "wrong-call"));
  }
}
