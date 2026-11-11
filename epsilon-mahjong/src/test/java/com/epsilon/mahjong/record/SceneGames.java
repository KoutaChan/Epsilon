package com.epsilon.mahjong.record;

import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;

/**
 * 牌山を手で並べて打った 1 局の半荘（{@link RecordedGames#scene}）のうち、自己対局では起きにくい場面: 包、ダブロン、槍槓、抜いた北
 * へのロン（ほかのモジュールのテストも使う）。終局と書いた局は、払った席が飛んで半荘が終わるので、半荘として再生できる。
 */
public final class SceneGames {

  private static final Variant FOUR = Variant.TENHOU_4P;
  private static final Variant THREE = Variant.TENHOU_3P;

  private SceneGames() {}

  /** 四人打ちの包のツモ。親の席 0 が大三元をツモり、3 つ目の三元牌（中）を鳴かせた席 1 が 48000 点を 1 人で払って飛ぶ（終局）。 */
  public static RecordedGame paoTsumo() {
    return thirdDragonFedBySeatOne(
        "pao-tsumo",
        "9s7z5m5m5s1p",
        "9s8m",
        "1:DAHAI_TSUMOGIRI(5m)",
        "2:DAHAI_TSUMOGIRI(5m)",
        "3:DAHAI_TSUMOGIRI(5s)",
        "0:TSUMO_AGARI");
  }

  /**
   * 四人打ちの、3 つ目の三元牌を鳴かせた席からのロン。親の席 0 が中のポンを加槓した後、席 1 の 1p にロンする。包の無いロンで、席 1 が
   * 48000 点を払って飛ぶ（終局）。
   */
  public static RecordedGame paoFeederRon() {
    return thirdDragonFedBySeatOne(
        "pao-feeder-ron",
        "9s7z5m5m5s7z1p",
        "9s8m9p",
        "1:DAHAI_TSUMOGIRI(5m)",
        "2:DAHAI_TSUMOGIRI(5m)",
        "3:DAHAI_TSUMOGIRI(5s)",
        "0:KAKAN(C)",
        "0:DAHAI_TSUMOGIRI(9p)",
        "1:DAHAI_TSUMOGIRI(1p)",
        "0:RON_AGARI");
  }

  /**
   * 四人打ちの、3 つ目の三元牌を鳴かせた席とは別の席からのロン。席 1 が白と發を暗槓し、席 2 が打った中をポンして大三元の聴牌になり、
   * 席 3 の 4p にロンする。放銃した席 3 と包の席 2 が 16000 点ずつ払う。
   */
  public static RecordedGame paoSplitRon() {
    return RecordedGames.scene(
        Scenes.first(FOUR)
            .hand(0, "147m258p369s1234z")
            .hand(1, "5555z6666z77z1m23p")
            .hand(2, "258m369p147s1234z")
            .hand(3, "369m147p258s1234z")
            .draws("9m9s7z1p4p9m1p")
            .rinshan("9s8m"),
        "pao-split-ron",
        "0:DAHAI_TSUMOGIRI(9m)",
        "1:ANKAN(P)",
        "1:ANKAN(F)",
        "1:DAHAI(1m)",
        "2:DAHAI_TSUMOGIRI(C)",
        "1:PON(C)",
        "1:DAHAI(8m)",
        "2:DAHAI(W)",
        "3:DAHAI_TSUMOGIRI(4p)",
        "1:RON_AGARI");
  }

  /**
   * 四人打ちのダブロン（東 2 局、供託 1 本）。親の席 1 が打った 5p に、席 2（一気通貫）と席 3（断么九）がロンする。
   *
   * @param honba 本場
   */
  public static RecordedGame doubleRon(int honba) {
    return RecordedGames.scene(
        Scenes.at(FOUR, 1, honba, 1, 25_000, 25_000, 24_000, 25_000)
            .hand(1, "5p147m147s369m123z")
            .hand(2, "123m456m789m46p11z")
            .hand(3, "234s567s678m34p55s")
            .hand(0, "19p258m1479s4567z")
            .draws("4z"),
        "double-ron-" + honba,
        "1:DAHAI(5p)",
        "2:RON_AGARI",
        "3:RON_AGARI");
  }

  /** 三人打ちのダブロン（東 1 局 2 本場、供託 1 本）。親の席 0 がツモ切った 5p に、席 1 と席 2 がロンする。 */
  public static RecordedGame threePlayerDoubleRon() {
    return RecordedGames.scene(
        Scenes.at(THREE, 0, 2, 1, 35_000, 35_000, 34_000)
            .hand(0, "19m258p147s369s67z")
            .hand(1, "123s456s789s46p11z")
            .hand(2, "234s678s678p34p55s")
            .draws("5p"),
        "double-ron-3p",
        "0:DAHAI_TSUMOGIRI(5p)",
        "1:RON_AGARI",
        "2:RON_AGARI");
  }

  /** 三人打ちの槍槓。親の席 0 が席 1 の 2p をポンし、4 枚目の 2p を加槓すると、34p で槍槓のほかに役の無い席 2 がロンする。 */
  public static RecordedGame chankan() {
    return RecordedGames.scene(
        Scenes.first(THREE)
            .hand(0, "22p19m58s9p123567z")
            .hand(1, "147p369s9m123567z")
            .hand(2, "34p111s789s456s99s")
            .draws("8p2p8p1z2p"),
        "chankan",
        "0:DAHAI(E)",
        "0:PON(2p)",
        "0:DAHAI(S)",
        "0:KAKAN(2p)",
        "2:RON_AGARI");
  }

  /**
   * 三人打ちの、抜いた北へのロン（東 1 局）。席 1 が 1 巡目に立直し、席 0 が抜いた北にロンする。
   *
   * @param scores 局の始まりの点（絶対席の順）
   */
  public static RecordedGame ronOnNuki(int... scores) {
    return RecordedGames.scene(
        Scenes.at(THREE, 0, 0, 0, scores)
            .hand(0, "19m258p369s147s5z6z")
            .hand(1, "123p456p789s11z44z")
            .hand(2, "19m369p258s147s3z7z")
            .draws("9p2z8p4z")
            .rinshan("5s"),
        "ron-on-nuki",
        "1:RIICHI_TSUMOGIRI(S)",
        "0:NUKI",
        "1:RON_AGARI");
  }

  /**
   * 四人打ちの、親の席 0 が白と發を暗槓し、席 1 がツモ切った中をポンして大三元の聴牌（1p・4p 待ち）になる局。
   *
   * @param id 牌譜 id
   * @param draws 山のツモ（局全体のツモの順）
   * @param rinshan 嶺上のツモ
   * @param rest 中のポンの後の打牌に続く選び
   */
  private static RecordedGame thirdDragonFedBySeatOne(String id, String draws, String rinshan, String... rest) {
    List<String> picks =
        new ArrayList<>(
            List.of("0:ANKAN(P)", "0:ANKAN(F)", "0:DAHAI(1m)", "1:DAHAI_TSUMOGIRI(C)", "0:PON(C)", "0:DAHAI(8m)"));
    picks.addAll(List.of(rest));
    return RecordedGames.scene(
        Scenes.first(FOUR)
            .hand(0, "5555z6666z77z1m23p")
            .hand(1, "147p258s369s1234z")
            .hand(2, "258p147s369m1234z")
            .hand(3, "369p258s147m1234z")
            .draws(draws)
            .rinshan(rinshan),
        id,
        picks.toArray(String[]::new));
  }
}
