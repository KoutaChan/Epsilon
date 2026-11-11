package com.epsilon.mahjong.score;

/**
 * 採点の結果。{@link HandScorer} が書き込む、使い回す可変のオブジェクトで、同時に使えるのは 1 スレッドだけ。役が無ければ全て 0。
 * 役満なら役には役満だけが入り、翻・符・ドラ・赤ドラ・裏ドラ・抜きドラは 0。支払いは、基本点から {@code ScoreMath} と
 * {@code Settlement} で求める。
 */
public final class WinScore {

  private long yaku;
  private boolean closed;
  private int han;
  private int fu;
  private int dora;
  private int aka;
  private int ura;
  private int nuki;
  private int yakuman;
  private int base;

  public void clear() {
    set(0, false, 0, 0, 0, 0, 0, 0, 0, 0);
  }

  public void copyFrom(WinScore other) {
    set(other.yaku, other.closed, other.han, other.fu, other.dora, other.aka, other.ura, other.nuki, other.yakuman,
        other.base);
  }

  /** 役の集合。役ごとのビット（{@link ScoringYaku#bit}）の論理和で表す。 */
  public long yaku() {
    return yaku;
  }

  public boolean has(ScoringYaku yaku) {
    return (this.yaku & yaku.bit()) != 0;
  }

  /** 門前の手か（暗槓だけの手も門前）。役の翻は、この値を {@link ScoringYaku#han} に渡して求める。 */
  public boolean closed() {
    return closed;
  }

  /** 役の翻にドラ・赤ドラ・裏ドラ・抜きドラを足した翻。 */
  public int han() {
    return han;
  }

  public int fu() {
    return fu;
  }

  /** ドラの翻。純手牌・面子・抜いた北から数える。 */
  public int dora() {
    return dora;
  }

  public int aka() {
    return aka;
  }

  /** 裏ドラの翻（立直した和了だけ）。ドラと同じく、純手牌・面子・抜いた北から数える。 */
  public int ura() {
    return ura;
  }

  /** 抜きドラの翻。抜いた北 1 枚につき 1 翻。北がドラや裏ドラのときに抜いた北で増える分は、ドラと裏ドラに入る。 */
  public int nuki() {
    return nuki;
  }

  /** 役満の倍率。役満でなければ 0。13 翻以上の数え役満も 0 で、基本点だけが役満と同じ額になる。 */
  public int yakuman() {
    return yakuman;
  }

  /** 基本点（{@code ScoreMath.base} の値）。 */
  public int base() {
    return base;
  }

  void set(long yaku, boolean closed, int han, int fu, int dora, int aka, int ura, int nuki, int yakuman, int base) {
    this.yaku = yaku;
    this.closed = closed;
    this.han = han;
    this.fu = fu;
    this.dora = dora;
    this.aka = aka;
    this.ura = ura;
    this.nuki = nuki;
    this.yakuman = yakuman;
    this.base = base;
  }

  @Override
  public String toString() {
    StringBuilder text = new StringBuilder("WinScore[");
    if (yakuman > 0) {
      text.append("yakuman=").append(yakuman);
    } else {
      text.append("han=").append(han).append(" fu=").append(fu);
    }
    text.append(" base=").append(base).append(closed ? " closed" : " open").append(" yaku=[");
    String separator = "";
    for (ScoringYaku each : ScoringYaku.values()) {
      if (has(each)) {
        text.append(separator).append(each);
        separator = ", ";
      }
    }
    text.append(']');
    if (yakuman == 0) {
      text.append(" dora=").append(dora).append(" aka=").append(aka).append(" ura=").append(ura)
          .append(" nuki=").append(nuki);
    }
    return text.append(']').toString();
  }
}
