package com.epsilon.calculate.scoring;

/** 天鳳4人打ちで採用する役。ドラは役ではないためここには含めない。 */
public enum ScoringYaku {
  /** 立直。 */
  RIICHI("立直", 1, 0),
  /** ダブル立直。 */
  DOUBLE_RIICHI("ダブル立直", 2, 0),
  /** 一発。 */
  IPPATSU("一発", 1, 0),
  /** 門前清自摸和。 */
  MENZEN_TSUMO("門前清自摸和", 1, 0),
  /** 平和。 */
  PINFU("平和", 1, 0),
  /** 断么九。 */
  TANYAO("断么九", 1, 1),
  /** 白の役牌。 */
  YAKUHAI_HAKU("役牌:白", 1, 1),
  /** 發の役牌。 */
  YAKUHAI_HATSU("役牌:發", 1, 1),
  /** 中の役牌。 */
  YAKUHAI_CHUN("役牌:中", 1, 1),
  /** 自風の役牌。 */
  YAKUHAI_SEAT("自風", 1, 1),
  /** 場風の役牌。 */
  YAKUHAI_ROUND("場風", 1, 1),
  /** 嶺上開花。 */
  RINSHAN("嶺上開花", 1, 1),
  /** 槍槓。 */
  CHANKAN("槍槓", 1, 1),
  /** 海底摸月。 */
  HAITEI("海底摸月", 1, 1),
  /** 河底撈魚。 */
  HOUTEI("河底撈魚", 1, 1),
  /** 一盃口。 */
  IPEIKOU("一盃口", 1, 0),
  /** 三色同順。 */
  SANSHOKU("三色同順", 2, 1),
  /** 三色同刻。 */
  SANSHOKU_DOUKOU("三色同刻", 2, 2),
  /** 一気通貫。 */
  ITTSU("一気通貫", 2, 1),
  /** 対々和。 */
  TOITOI("対々和", 2, 2),
  /** 三暗刻。 */
  SANANKOU("三暗刻", 2, 2),
  /** 三槓子。 */
  SANKANTSU("三槓子", 2, 2),
  /** 小三元。 */
  SHOUSANGEN("小三元", 2, 2),
  /** 混老頭。 */
  HONROUTOU("混老頭", 2, 2),
  /** 混全帯么九。 */
  CHANTA("全帯幺", 2, 1),
  /** 二盃口。 */
  RYANPEIKOU("二盃口", 3, 0),
  /** 純全帯么九。 */
  JUNCHAN("純全帯幺", 3, 2),
  /** 混一色。 */
  HONITSU("混一色", 3, 2),
  /** 清一色。 */
  CHINITSU("清一色", 6, 5),
  /** 七対子。 */
  CHIITOITSU("七対子", 2, 0),
  /** 国士無双。 */
  KOKUSHI("国士無双"),
  /** 四暗刻。 */
  SUUANKOU("四暗刻"),
  /** 九蓮宝燈。 */
  CHUUREN("九蓮宝燈"),
  /** 大三元。 */
  DAISANGEN("大三元"),
  /** 字一色。 */
  TSUUIISOU("字一色"),
  /** 緑一色。 */
  RYUUIISOU("緑一色"),
  /** 清老頭。 */
  CHINROUTOU("清老頭"),
  /** 大四喜。 */
  DAISUUSHII("大四喜"),
  /** 小四喜。 */
  SHOUSUUSHII("小四喜"),
  /** 四槓子。 */
  SUUKANTSU("四槓子"),
  /** 天和。 */
  TENHOU("天和"),
  /** 地和。 */
  CHIIHOU("地和");

  private final String label;
  private final int menzenHan;
  private final int openHan;
  private final int yakumanMultiplier;

  ScoringYaku(String label, int menzenHan, int openHan) {
    this.label = label;
    this.menzenHan = menzenHan;
    this.openHan = openHan;
    this.yakumanMultiplier = 0;
  }

  ScoringYaku(String label) {
    this.label = label;
    this.menzenHan = 0;
    this.openHan = 0;
    this.yakumanMultiplier = 1;
  }

  public long bit() {
    return 1L << ordinal();
  }

  /**
   * 牌譜や表示に使う日本語の役名を返す。
   *
   * @return 日本語の役名
   */
  public String label() {
    return label;
  }

  /**
   * 門前・副露状態に対応する翻数を返す。
   *
   * @param menzen 門前なら {@code true}
   * @return 翻数。成立しない門前・副露の状態または役満では0
   */
  public int han(boolean menzen) {
    return menzen ? menzenHan : openHan;
  }

  /**
   * 通常役ではなく役満かを返す。
   *
   * @return 役満なら {@code true}
   */
  public boolean isYakuman() {
    return yakumanMultiplier > 0;
  }

  /** この役の役満倍率。天鳳四人打ちでは各役満を一倍として加算する。 */
  public int yakumanMultiplier() {
    return yakumanMultiplier;
  }
}
