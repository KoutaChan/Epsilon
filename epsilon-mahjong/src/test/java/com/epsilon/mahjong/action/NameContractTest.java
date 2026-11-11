package com.epsilon.mahjong.action;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * PRISM の語彙で使う名前を確かめる。フィンガープリントには、パッケージを除いた型の名前（Outer$Inner）と、語彙の並びを書いた
 * 文字列が入る。並びには、そのルールで使う定数だけを宣言の順に入れる（四人打ちでは NUKI が、三人打ちでは CHI が抜ける）。
 * 四人打ちの並びの文字列は系譜のフィンガープリントに入っているので、文字列そのものを比べる。
 */
public class NameContractTest {

  @Test
  public void fourPlayerVocabulariesKeepTheirText() {
    Variant variant = Variant.TENHOU_4P;
    assertText(
        Action.Type.class,
        variant,
        "[DAHAI, CHI, PON, DAIMINKAN, ANKAN, KAKAN, RIICHI_DAHAI, TSUMO_AGARI, RON_AGARI, PASS, KYUSHU_KYUHAI]");
    assertText(Action.TileSelection.class, variant, "[NOT_APPLICABLE, NON_AKA, AKA, TSUMOGIRI, TSUMOGIRI_AKA]");
    assertText(Meld.Type.class, variant, "[CHI, PON, DAIMINKAN, ANKAN, KAKAN]");
    assertText(Meld.RelativeSource.class, variant, "[NONE, SHIMOCHA, TOIMEN, KAMICHA]");
    assertText(Meld.AkaSource.class, variant, "[NONE, CALLED_TILE, CONSUMED_HAND_TILE, ADDED_KAN_TILE]");
  }

  @Test
  public void threePlayerVocabulariesDropChiAndKeepTheOrder() {
    Variant variant = Variant.TENHOU_3P;
    assertText(
        Action.Type.class,
        variant,
        "[DAHAI, PON, DAIMINKAN, ANKAN, KAKAN, RIICHI_DAHAI, TSUMO_AGARI, RON_AGARI, PASS, KYUSHU_KYUHAI, NUKI]");
    assertText(Action.TileSelection.class, variant, "[NOT_APPLICABLE, NON_AKA, AKA, TSUMOGIRI, TSUMOGIRI_AKA]");
    assertText(Meld.Type.class, variant, "[PON, DAIMINKAN, ANKAN, KAKAN]");
    assertText(Meld.RelativeSource.class, variant, "[NONE, SHIMOCHA, TOIMEN, KAMICHA]");
    assertText(Meld.AkaSource.class, variant, "[NONE, CALLED_TILE, CONSUMED_HAND_TILE, ADDED_KAN_TILE]");
    Vocabulary<Action.Type> types = Vocabulary.of(Action.Type.class, variant);
    Assert.assertEquals(types.index(Action.Type.PON), 1);
    Assert.assertEquals(types.index(Action.Type.NUKI), 10);
  }

  @Test
  public void digestNamesKeepTheOuterClasses() {
    Assert.assertEquals(digestName(Action.Type.class), "Action$Type");
    Assert.assertEquals(digestName(Action.TileSelection.class), "Action$TileSelection");
    Assert.assertEquals(digestName(Meld.Type.class), "Meld$Type");
    Assert.assertEquals(digestName(Meld.RelativeSource.class), "Meld$RelativeSource");
    Assert.assertEquals(digestName(Meld.AkaSource.class), "Meld$AkaSource");
  }

  private static <E extends Enum<E>> void assertText(Class<E> type, Variant variant, String expected) {
    Assert.assertEquals(Vocabulary.of(type, variant).toString(), expected, type.getName() + " " + variant);
  }

  private static String digestName(Class<?> type) {
    return type.getName().substring(type.getPackageName().length() + 1);
  }
}
