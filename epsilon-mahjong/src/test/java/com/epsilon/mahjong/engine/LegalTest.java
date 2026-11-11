package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.RelativeSource;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.Tiles;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 手番と鳴いた後の打牌で、合法手を確かめる。見るのは、決まった並び順、槓・立直・九種九牌・喰い替えの条件、鳴いた面子の番号と赤の出どころ。 */
public class LegalTest {

  /** 門前の手番の合法手は、ツモ和了 → 暗槓（種類の昇順）→ 立直の打牌 → 打牌の順。同じ種類では、ツモ切りが後に並ぶ。 */
  @Test
  public void aClosedTurnListsTheWinKansRiichiAndDiscardsInOrder() {
    Scene scene = Scene.first().river(0, "9s").hand(0, "1111m23m2222p34p5s").draw(0, "5s").turn();
    Assert.assertEquals(
        scene.legal(0),
        List.of(
            "TSUMO_AGARI",
            "ANKAN(1m)",
            "ANKAN(2p)",
            "RIICHI_DAHAI(1m)",
            "RIICHI_DAHAI(2m)",
            "RIICHI_DAHAI(3m)",
            "RIICHI_DAHAI(2p)",
            "RIICHI_DAHAI(3p)",
            "RIICHI_DAHAI(4p)",
            "RIICHI_DAHAI(5s)",
            "RIICHI_TSUMOGIRI(5s)",
            "DAHAI(1m)",
            "DAHAI(2m)",
            "DAHAI(3m)",
            "DAHAI(2p)",
            "DAHAI(3p)",
            "DAHAI(4p)",
            "DAHAI(5s)",
            "DAHAI_TSUMOGIRI(5s)"));
    Assert.assertEquals(scene.decisions.size(), 1);
    Assert.assertEquals(scene.decisions.kind(0), EngineDecisionKind.TURN);
    Assert.assertEquals(scene.decisions.id(0), 1L);
    Assert.assertEquals(scene.nextId(), 2L);
    Assert.assertTrue(scene.decisions.win(0).has(ScoringYaku.MENZEN_TSUMO));
  }

  /** 副露した手番の合法手は、加槓（面子の順）→ 打牌の順。立直は出ない。 */
  @Test
  public void anOpenTurnListsKakanThenDiscards() {
    Scene scene = Scene.first().pon(1, 0, "7z").hand(1, "7z123m456p789s").draw(1, "1z").turn();
    Assert.assertEquals(
        scene.legal(1),
        List.of(
            "KAKAN(C)",
            "DAHAI(1m)",
            "DAHAI(2m)",
            "DAHAI(3m)",
            "DAHAI(4p)",
            "DAHAI(5p)",
            "DAHAI(6p)",
            "DAHAI(7s)",
            "DAHAI(8s)",
            "DAHAI(9s)",
            "DAHAI_TSUMOGIRI(E)",
            "DAHAI(C)"));
  }

  /** 同じ種類の打牌は、手牌の赤 → 手牌の赤でない牌 → ツモ切りの順。ツモった牌が赤なら、ツモ切りが赤になる。 */
  @Test
  public void redFivesComeFirstAndTheDrawnTileLast() {
    Scene plain = Scene.first().pon(1, 0, "7z").hand(1, "05m123p456p99s").draw(1, "5m").turn();
    Assert.assertEquals(
        plain.legal(1),
        List.of(
            "TSUMO_AGARI",
            "DAHAI_AKA(5mr)",
            "DAHAI(5m)",
            "DAHAI_TSUMOGIRI(5m)",
            "DAHAI(1p)",
            "DAHAI(2p)",
            "DAHAI(3p)",
            "DAHAI(4p)",
            "DAHAI(5p)",
            "DAHAI(6p)",
            "DAHAI(9s)"));
    Scene red = Scene.first().pon(1, 0, "7z").hand(1, "55m123p456p99s").draw(1, "0m").turn();
    Assert.assertEquals(red.legal(1).subList(0, 3), List.of("TSUMO_AGARI", "DAHAI(5m)", "DAHAI_TSUMOGIRI_AKA(5mr)"));
    Assert.assertTrue(red.decisions.win(0).has(ScoringYaku.YAKUHAI_CHUN));
    Assert.assertEquals(red.decisions.win(0).aka(), 1);
  }

  /** 九種九牌は、鳴きや槓で巡が途切れていない局での自分の最初のツモで、么九牌が 9 種以上あるときだけ出る。並びの最後に置く。 */
  @Test
  public void nineTerminalsNeedNineKindsOnAnUninterruptedFirstDraw() {
    List<String> nine = Scene.first().hand(0, "19m19p19s123z2345m").draw(0, "6m").turn().legal(0);
    Assert.assertEquals(nine.get(nine.size() - 1), "KYUSHU_KYUHAI");
    Assert.assertFalse(
        Scene.first().hand(0, "19m19p19s12z23456m").draw(0, "7m").turn().legal(0).contains("KYUSHU_KYUHAI"));
    Assert.assertFalse(
        Scene.first()
            .ankan(2, "5z")
            .hand(0, "19m19p19s123z2345m")
            .draw(0, "6m")
            .turn()
            .legal(0)
            .contains("KYUSHU_KYUHAI"));
    Assert.assertFalse(
        Scene.first()
            .river(0, "7z")
            .hand(0, "19m19p19s123z2345m")
            .draw(0, "6m")
            .turn()
            .legal(0)
            .contains("KYUSHU_KYUHAI"));
  }

  /** 立直できるのは、門前（暗槓は門前を崩さない）で、1000 点以上あり、山の残りがルールの下限以上のときだけ。 */
  @Test
  public void riichiNeedsAClosedHandAThousandPointsAndEnoughWall() {
    Assert.assertTrue(riichi(Scene.first()));
    Assert.assertTrue(riichi(Scene.withScores(1_000, 33_000, 33_000, 33_000)));
    Assert.assertFalse(riichi(Scene.withScores(900, 33_100, 33_000, 33_000)));
    int minWall = Scene.VARIANT.riichiMinWall();
    int liveWall = Scene.VARIANT.liveWall();
    Assert.assertTrue(riichi(Scene.first().burn(liveWall - minWall - 1)));
    Assert.assertFalse(riichi(Scene.first().burn(liveWall - minWall)));
    Assert.assertFalse(
        Scene.first().pon(0, 1, "7z").hand(0, "123m456p789s1z").draw(0, "2z").turn().legal(0).stream()
            .anyMatch(name -> name.startsWith("RIICHI")));
    Assert.assertEquals(
        Scene.first().ankan(0, "5z").hand(0, "123m456p789s1z").draw(0, "2z").turn().legal(0).stream()
            .filter(name -> name.startsWith("RIICHI"))
            .toList(),
        List.of("RIICHI_DAHAI(E)", "RIICHI_TSUMOGIRI(S)"));
  }

  /** 槓できるのは、場の槓が 4 つ未満で、山の残りが 1 枚以上のときだけ。 */
  @Test
  public void kansNeedFewerThanFourKansAndATileLeft() {
    Assert.assertTrue(quad(Scene.first().ankan(1, "1z").ankan(2, "2z").ankan(3, "3z")).contains("ANKAN(1m)"));
    Assert.assertFalse(
        quad(Scene.first().ankan(1, "1z").ankan(2, "2z").ankan(3, "3z").ankan(1, "4z")).contains("ANKAN(1m)"));
    Assert.assertFalse(quad(Scene.first().burn(Scene.VARIANT.liveWall() - 1)).contains("ANKAN(1m)"));
  }

  /** 立直中の暗槓は、ツモった牌の種類で、待ちが変わらないときだけできる。暗槓も和了もできなければ、ツモ切りしかできない。 */
  @Test
  public void anAnkanInRiichiNeedsTheDrawnKindAndUnchangedWaits() {
    Assert.assertEquals(
        inRiichi("111m456p789s2355s").draw(0, "1m").turn().legal(0), List.of("ANKAN(1m)", "DAHAI_TSUMOGIRI(1m)"));
    Assert.assertEquals(inRiichi("3334m456p789s111z").draw(0, "3m").turn().legal(0), List.of("DAHAI_TSUMOGIRI(3m)"));
    Assert.assertEquals(inRiichi("1111m23m456p789s5s").draw(0, "7z").turn().legal(0), List.of("DAHAI_TSUMOGIRI(C)"));
    Assert.assertEquals(
        inRiichi("1111m23m456p789s5s").draw(0, "0s").turn().legal(0),
        List.of("TSUMO_AGARI", "DAHAI_TSUMOGIRI_AKA(5sr)"));
  }

  /** 鳴いた後の打牌は、喰い替えになる牌（ポンは鳴いた種類、チーは鳴いた種類と筋の喰い替えになる種類）を除き、手番の打牌と同じ順に並べる。 */
  @Test
  public void postCallDiscardsLeaveOutTheSwapBan() {
    Scene chi = Scene.first().hand(1, "2345m6899p789s11z").river(0, "2m").chi(1, "234m").postCall(1);
    Assert.assertEquals(chi.decisions.kind(0), EngineDecisionKind.POST_CALL_DAHAI);
    Assert.assertEquals(
        chi.legal(1),
        List.of("DAHAI(6p)", "DAHAI(8p)", "DAHAI(9p)", "DAHAI(7s)", "DAHAI(8s)", "DAHAI(9s)", "DAHAI(E)"));
    Scene pon = Scene.first().hand(1, "055p123m789s1122z").river(0, "5p").ponDiscard(1, false).postCall(1);
    Assert.assertEquals(
        pon.legal(1),
        List.of("DAHAI(1m)", "DAHAI(2m)", "DAHAI(3m)", "DAHAI(7s)", "DAHAI(8s)", "DAHAI(9s)", "DAHAI(E)", "DAHAI(S)"));
    Scene red = Scene.first().hand(1, "05568m123p789s11z").river(0, "7m").chi(1, "678m").postCall(1);
    Assert.assertEquals(red.legal(1).subList(0, 2), List.of("DAHAI_AKA(5mr)", "DAHAI(5m)"));
  }

  @Test
  public void swapBansCoverTheCalledKindAndTheOuterSuji() {
    Assert.assertEquals(Legal.swapBan(chi(Tiles.M2, Tiles.M2)), Tiles.bit(Tiles.M2) | Tiles.bit(Tiles.M5));
    Assert.assertEquals(Legal.swapBan(chi(Tiles.M2, Tiles.M3)), Tiles.bit(Tiles.M3));
    Assert.assertEquals(Legal.swapBan(chi(Tiles.M2, Tiles.M4)), Tiles.bit(Tiles.M4) | Tiles.bit(Tiles.M1));
    Assert.assertEquals(Legal.swapBan(chi(Tiles.P7, Tiles.P7)), Tiles.bit(Tiles.P7));
    Assert.assertEquals(Legal.swapBan(chi(Tiles.S1, Tiles.S3)), Tiles.bit(Tiles.S3));
    Assert.assertEquals(
        Legal.swapBan(Meld.pon(Tiles.P5, RelativeSource.TOIMEN, AkaSource.NONE)), Tiles.bit(Tiles.P5));
    Assert.assertThrows(IllegalArgumentException.class, () -> Legal.swapBan(Meld.ankan(Tiles.P1, false)));
  }

  /**
   * 鳴いた面子に入る赤の出どころを確かめる。鳴いた牌が赤なら、出どころは鳴いた牌。そうでなければ、チー・ポンでは手牌の赤を使う手を
   * 選んだかで決まり、大明槓では手牌に赤があるかで決まる。
   */
  @Test
  public void callMeldsTakeTheRedSourceOfTheCall() {
    SeatView fromHand = Scene.first().hand(1, "055p123m789s1122z").river(0, "5p").full.seat(1);
    Assert.assertEquals(
        callMeld(fromHand, Action.code(Type.PON, Tiles.P5, TileSelection.AKA)),
        Meld.pon(Tiles.P5, RelativeSource.KAMICHA, AkaSource.CONSUMED_HAND_TILE));
    Assert.assertEquals(
        callMeld(fromHand, Action.code(Type.PON, Tiles.P5, TileSelection.NON_AKA)),
        Meld.pon(Tiles.P5, RelativeSource.KAMICHA, AkaSource.NONE));
    Assert.assertEquals(
        callMeld(fromHand, Action.code(Type.DAIMINKAN, Tiles.P5, TileSelection.NOT_APPLICABLE)),
        Meld.daiminkan(Tiles.P5, RelativeSource.KAMICHA, AkaSource.CONSUMED_HAND_TILE));

    SeatView called = Scene.first().hand(2, "555p123m789s1122z").river(0, "0p").full.seat(2);
    Assert.assertEquals(
        callMeld(called, Action.code(Type.PON, Tiles.P5, TileSelection.NON_AKA)),
        Meld.pon(Tiles.P5, RelativeSource.TOIMEN, AkaSource.CALLED_TILE));
    Assert.assertEquals(
        callMeld(called, Action.code(Type.DAIMINKAN, Tiles.P5, TileSelection.NOT_APPLICABLE)),
        Meld.daiminkan(Tiles.P5, RelativeSource.TOIMEN, AkaSource.CALLED_TILE));

    SeatView chi = Scene.first().hand(1, "069p44p123m789s12z").river(0, "4p").full.seat(1);
    Assert.assertEquals(
        callMeld(chi, Action.chi(Tiles.P4, Tiles.P4, true)),
        Meld.chi(Tiles.P4, Tiles.P4, AkaSource.CONSUMED_HAND_TILE));
    Assert.assertEquals(
        callMeld(chi, Action.code(Type.PON, Tiles.P4, TileSelection.NON_AKA)),
        Meld.pon(Tiles.P4, RelativeSource.KAMICHA, AkaSource.NONE));
  }

  /**
   * 他家の打牌の直後でなければ IllegalStateException を投げる。鳴きでない手、打牌と種類の違う鳴き、上家でない席のチー、手牌に足りない牌
   * （赤を使う選び方での赤と、赤を使わない選び方での赤でない牌を含む）には IllegalArgumentException を投げる。
   */
  @Test
  public void callMeldRejectsCallsTheDiscardOrTheHandCannotMake() {
    int ponFour = Action.code(Type.PON, Tiles.P4, TileSelection.NON_AKA);
    SeatView own = Scene.first().river(0, "4p").full.seat(0);
    Assert.assertThrows(IllegalStateException.class, () -> Legal.callMeld(own, ponFour));
    SeatView drawn = Scene.first().draw(1, "4p").full.seat(1);
    Assert.assertThrows(IllegalStateException.class, () -> Legal.callMeld(drawn, ponFour));

    SeatView chi = Scene.first().hand(1, "069p44p123m789s12z").river(0, "4p").full.seat(1);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Legal.callMeld(chi, Action.code(Type.DAHAI, Tiles.P4, TileSelection.NON_AKA)));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Legal.callMeld(chi, Action.code(Type.PON, Tiles.P6, TileSelection.NON_AKA)));
    Assert.assertThrows(IllegalArgumentException.class, () -> Legal.callMeld(chi, Action.chi(Tiles.P4, Tiles.P4, false)));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Legal.callMeld(chi, Action.code(Type.DAIMINKAN, Tiles.P4, TileSelection.NOT_APPLICABLE)));

    SeatView toimen = Scene.first().hand(2, "46p123m789s11223z").river(0, "5p").full.seat(2);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Legal.callMeld(toimen, Action.chi(Tiles.P4, Tiles.P5, false)));

    SeatView plain = Scene.first().hand(1, "55p123m789s11223z").river(0, "5p").full.seat(1);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Legal.callMeld(plain, Action.code(Type.PON, Tiles.P5, TileSelection.AKA)));
  }

  private static Meld callMeld(SeatView view, int code) {
    return Meld.of(Legal.callMeld(view, code));
  }

  /** 席 0 が 9p を宣言牌にして立直した手。 */
  private static Scene inRiichi(String hand) {
    return Scene.first().hand(0, hand).river(0, "9p").riichi(0);
  }

  /** 123m456p789s1122z の手に 3z をツモったとき、3z をツモ切りする立直が合法手に出るか。 */
  private static boolean riichi(Scene scene) {
    return scene.hand(0, "123m456p789s1122z").draw(0, "3z").turn().legal(0).contains("RIICHI_TSUMOGIRI(W)");
  }

  /** 1m を 4 枚持つ手に 5z をツモったときの合法手。 */
  private static List<String> quad(Scene scene) {
    return scene.hand(0, "1111m234p567p789s").draw(0, "5z").turn().legal(0);
  }

  private static Meld chi(int base, int called) {
    return Meld.chi(base, called, AkaSource.NONE);
  }
}
