package com.epsilon.classic.rulebot;

/**
 * 合法手を選んだ後の自分の手の値。{@link Projection} が書き、{@link Rules} が比べる。判断のたびに作業領域の中で使い回す。
 *
 * <p>受け入れと待ちの枚数は、判断の時点で自分から見えていない枚数で数える。待ちの値は、役のある待ちだけを数える。ロンの点数は放銃者が
 * 払う点で、本場・供託・裏ドラは含まない。
 */
final class Quality {

  /** 良い待ちと見る、役のある待ちの生きている枚数の下限。 */
  private static final int GOOD_WAIT_TILES = 5;

  /** 良い待ちと見る、役のある待ちの種類の数の下限。 */
  private static final int GOOD_WAIT_KINDS = 2;

  private int shanten;
  private boolean special;
  private int chiitoitsuShanten;
  private int ukeire;
  private int ukeireKinds;
  private int secondOrder;
  private long waits;
  private boolean allSimple;
  private boolean oneSuit;
  private int dora;
  private int aka;
  private int confirmedYaku;
  private int liveWaits;
  private int liveWaitKinds;
  private int liveRon;
  private int minimumRon;
  private int averageRon;

  /**
   * 手の形を書く。
   *
   * @param shanten 通常形・七対子・国士のうち最小の向聴
   * @param special 面子が無く、七対子と国士も数える手なら true
   * @param chiitoitsuShanten 七対子の向聴（面子のある手は読まない）
   * @param ukeire 受け入れの生きている枚数
   * @param ukeireKinds 受け入れのうち、生きている牌が残る種類の数
   * @param secondOrder 2 枚目の受け入れ。受け入れの牌を 1 枚引いていちばん良い牌を打った後の受け入れの枚数を、引く牌の生きている
   *     枚数で重みを付けて平均した値（1・2 向聴の手だけ。ほかは 0）
   * @param waits 形の上の和了牌の種類のビット（聴牌の手だけ）
   */
  void setShape(
      int shanten, boolean special, int chiitoitsuShanten, int ukeire, int ukeireKinds, int secondOrder, long waits) {
    this.shanten = shanten;
    this.special = special;
    this.chiitoitsuShanten = chiitoitsuShanten;
    this.ukeire = ukeire;
    this.ukeireKinds = ukeireKinds;
    this.secondOrder = secondOrder;
    this.waits = waits;
  }

  /**
   * 手の牌と面子から決まる値を書く。
   *
   * @param allSimple 純手牌と面子に么九牌が無ければ true
   * @param oneSuit 純手牌と面子の数牌が 1 色以下なら true
   * @param dora 純手牌と面子のドラの数（同じ牌を指す表示牌が複数あれば、その数だけ数える）
   * @param aka 純手牌と面子の赤 5 の数
   * @param confirmedYaku 面子だけで決まった役の数。役牌の刻子・槓子は付く役の数だけ数える。ほかに、面子 4 つにチーが無いときの対々和と、
   *     鳴いた順子だけでできる一気通貫・三色同順を数える
   */
  void setHand(boolean allSimple, boolean oneSuit, int dora, int aka, int confirmedYaku) {
    this.allSimple = allSimple;
    this.oneSuit = oneSuit;
    this.dora = dora;
    this.aka = aka;
    this.confirmedYaku = confirmedYaku;
  }

  /**
   * 待ちの値を書く。聴牌でない手では全て 0。
   *
   * @param liveWaits ロンかツモで役のある待ちの生きている枚数
   * @param liveWaitKinds ロンかツモで役のある待ちのうち、生きている牌が残る種類の数
   * @param liveRon ロンで役のある待ちの生きている枚数
   * @param minimumRon ロンで役のある待ちのうち、いちばん安い点数（無ければ 0）
   * @param averageRon ロンの点数を、生きている枚数で重みを付けて平均した値（無ければ 0）
   */
  void setWaits(int liveWaits, int liveWaitKinds, int liveRon, int minimumRon, int averageRon) {
    this.liveWaits = liveWaits;
    this.liveWaitKinds = liveWaitKinds;
    this.liveRon = liveRon;
    this.minimumRon = minimumRon;
    this.averageRon = averageRon;
  }

  int shanten() {
    return shanten;
  }

  boolean special() {
    return special;
  }

  /** 七対子の向聴（{@link #special} の手だけが持つ）。 */
  int chiitoitsuShanten() {
    return chiitoitsuShanten;
  }

  int ukeire() {
    return ukeire;
  }

  int ukeireKinds() {
    return ukeireKinds;
  }

  int secondOrder() {
    return secondOrder;
  }

  long waits() {
    return waits;
  }

  boolean allSimple() {
    return allSimple;
  }

  boolean oneSuit() {
    return oneSuit;
  }

  int aka() {
    return aka;
  }

  int confirmedYaku() {
    return confirmedYaku;
  }

  int liveWaits() {
    return liveWaits;
  }

  int liveRon() {
    return liveRon;
  }

  int minimumRon() {
    return minimumRon;
  }

  int averageRon() {
    return averageRon;
  }

  /** 役に依らない翻（ドラと赤）。 */
  int valueHan() {
    return dora + aka;
  }

  /** 役のある待ちが 5 枚以上、2 種類以上残っているか。 */
  boolean goodWait() {
    return liveWaits >= GOOD_WAIT_TILES && liveWaitKinds >= GOOD_WAIT_KINDS;
  }
}
