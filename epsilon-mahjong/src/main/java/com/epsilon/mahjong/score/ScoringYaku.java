package com.epsilon.mahjong.score;

/**
 * 天鳳の役。ドラ・赤ドラ・裏ドラ・抜きドラは役に入れない。定数の名前と順は PRISM の語彙が使うので変えない。役の集合は long で
 * 持ち、k 番目のビットが k 番目（ordinal）の役を表す。
 *
 * <p>役満は 1 つにつき 1 倍で、複合すれば倍率を足す。天鳳は四暗刻単騎・国士十三面・純正九蓮・大四喜をダブル役満にしない。
 * そのため、それぞれ四暗刻・国士無双・九蓮宝燈・大四喜の 1 倍とする。
 */
public enum ScoringYaku {
  RIICHI(1, 0),
  DOUBLE_RIICHI(2, 0),
  /** 立直（ダブル立直を含む）してから一巡以内の和了。 */
  IPPATSU(1, 0),
  MENZEN_TSUMO(1, 0),
  PINFU(1, 0),
  TANYAO(1, 1),
  YAKUHAI_HAKU(1, 1),
  YAKUHAI_HATSU(1, 1),
  YAKUHAI_CHUN(1, 1),
  YAKUHAI_SEAT(1, 1),
  /** 連風牌の刻子には、自風の役と場風の役の両方が付く。 */
  YAKUHAI_ROUND(1, 1),
  /** 嶺上開花。槓の後の嶺上牌のツモと、抜きの後の補充のツモで付く。 */
  RINSHAN(1, 1),
  CHANKAN(1, 1),
  /** 嶺上牌のツモには付かない。 */
  HAITEI(1, 1),
  HOUTEI(1, 1),
  IPEIKOU(1, 0),
  SANSHOKU(2, 1),
  SANSHOKU_DOUKOU(2, 2),
  ITTSU(2, 1),
  TOITOI(2, 2),
  /** 暗槓も数える。ロンの双碰で完成した刻子は数えない。 */
  SANANKOU(2, 2),
  SANKANTSU(2, 2),
  SHOUSANGEN(2, 2),
  HONROUTOU(2, 2),
  /** 順子を 1 つ以上含み、字牌がある。順子が無ければ混老頭になる。 */
  CHANTA(2, 1),
  RYANPEIKOU(3, 0),
  /** 順子を 1 つ以上含み、字牌が無い。 */
  JUNCHAN(3, 2),
  HONITSU(3, 2),
  CHINITSU(6, 5),
  CHIITOITSU(2, 0),
  KOKUSHI,
  /** ロンの双碰で完成した刻子は暗刻に数えない。 */
  SUUANKOU,
  CHUUREN,
  DAISANGEN,
  TSUUIISOU,
  /** 發が無くてもよい。 */
  RYUUIISOU,
  CHINROUTOU,
  DAISUUSHII,
  SHOUSUUSHII,
  SUUKANTSU,
  /** 親が第一ツモで和了した。それまでに誰も鳴いておらず、嶺上牌のツモではない。 */
  TENHOU,
  /** 子が第一ツモで和了した。それまでに誰も鳴いておらず、嶺上牌のツモではない。 */
  CHIIHOU;

  public static final long YAKUMAN_BITS = yakumanBits();

  private final int closedHan;
  private final int openHan;
  private final boolean yakuman;

  ScoringYaku(int closedHan, int openHan) {
    this.closedHan = closedHan;
    this.openHan = openHan;
    yakuman = false;
  }

  ScoringYaku() {
    closedHan = 0;
    openHan = 0;
    yakuman = true;
  }

  public long bit() {
    return 1L << ordinal();
  }

  /**
   * この役に付く翻。門前の手か副露した手かで変わることがある。その手で付かない役と、役満は 0。
   *
   * @param closed 門前の手か
   * @return 翻
   */
  public int han(boolean closed) {
    return closed ? closedHan : openHan;
  }

  public boolean yakuman() {
    return yakuman;
  }

  private static long yakumanBits() {
    long bits = 0;
    for (ScoringYaku yaku : values()) {
      if (yaku.yakuman) {
        bits |= yaku.bit();
      }
    }
    return bits;
  }
}
