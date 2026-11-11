package com.epsilon.mahjong.score;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * PRISM が使う語彙（役と立直の状態）の名前と並びを固定する。モデルのフィンガープリントには、型の名前をパッケージを除いて書き、語彙は
 * 並びどおりの文字列で書く。四人打ちも三人打ちも全ての定数を持つ。学習済みの系譜のフィンガープリントと教師の列がこの名前と並びに
 * 依存するので、テストで固定する。
 */
public class YakuNameContractTest {

  private static final String YAKU =
      "[RIICHI, DOUBLE_RIICHI, IPPATSU, MENZEN_TSUMO, PINFU, TANYAO, YAKUHAI_HAKU, YAKUHAI_HATSU, YAKUHAI_CHUN,"
          + " YAKUHAI_SEAT, YAKUHAI_ROUND, RINSHAN, CHANKAN, HAITEI, HOUTEI, IPEIKOU, SANSHOKU, SANSHOKU_DOUKOU, ITTSU,"
          + " TOITOI, SANANKOU, SANKANTSU, SHOUSANGEN, HONROUTOU, CHANTA, RYANPEIKOU, JUNCHAN, HONITSU, CHINITSU,"
          + " CHIITOITSU, KOKUSHI, SUUANKOU, CHUUREN, DAISANGEN, TSUUIISOU, RYUUIISOU, CHINROUTOU, DAISUUSHII,"
          + " SHOUSUUSHII, SUUKANTSU, TENHOU, CHIIHOU]";

  @Test
  public void bothVariantsKeepTheVocabularyText() {
    for (Variant variant : Variant.values()) {
      Assert.assertEquals(Vocabulary.of(ScoringYaku.class, variant).toString(), YAKU, variant.name());
      Assert.assertEquals(
          Vocabulary.of(RiichiState.class, variant).toString(), "[NONE, RIICHI, DOUBLE_RIICHI]", variant.name());
    }
    Assert.assertEquals(ScoringYaku.values().length, 42);
  }

  /** 役のビットの位置は定数の順番（ordinal）で決まり、教師の列の k 番目は順番が k の役に当たる。 */
  @Test
  public void yakuBitsFollowTheOrdinal() {
    for (ScoringYaku yaku : ScoringYaku.values()) {
      Assert.assertEquals(yaku.bit(), 1L << yaku.ordinal(), yaku.name());
    }
  }

  @Test
  public void digestNamesAreTopLevel() {
    Assert.assertEquals(digestName(ScoringYaku.class), "ScoringYaku");
    Assert.assertEquals(digestName(RiichiState.class), "RiichiState");
  }

  private static String digestName(Class<?> type) {
    return type.getName().substring(type.getPackageName().length() + 1);
  }
}
