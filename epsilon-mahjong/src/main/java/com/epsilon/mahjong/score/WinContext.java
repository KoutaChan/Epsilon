package com.epsilon.mahjong.score;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Objects;

/**
 * 採点に要る入力を全て持つ。{@link HandScorer} はこれだけを読み、エンジンの状態は読まない。使い回す可変のオブジェクトで、ルールは
 * 作るときに決まる。同時に使えるのは 1 スレッドだけ。牌の種類は {@code Tiles} の番号で表す。
 *
 * <p>純手牌（副露・暗槓・抜いた北を除く）は {@link #hand} に書き、面子は {@link #addMeld} で足す。{@link #hand} に書く面子の
 * 数は、足した面子の数と同じにする。手の枚数（面子 1 つを 3 枚と数える）が 13 なら、和了牌を足して採点する。14 なら、和了牌を
 * 含む手として採点する。値を書くメソッドは、その値の範囲を確かめる。手・面子・赤ドラ・抜きドラ・和了の条件の間に矛盾が無いかは、
 * 採点のときに確かめる。
 */
public final class WinContext {

  /** ドラ表示牌の最大の枚数（最初の 1 枚と槓ドラ）。 */
  public static final int MAX_INDICATORS = 1 + Variant.MAX_KANS;

  /** 和了牌の出どころ。 */
  public enum Source {
    TSUMO(true, null),
    /** 和了した人が槓か抜きをした後の、嶺上牌か補充の牌のツモ。山の最後の牌でも海底は付かない。 */
    RINSHAN(true, ScoringYaku.RINSHAN),
    /** 山の最後の牌のツモ。 */
    HAITEI(true, ScoringYaku.HAITEI),
    /** 打牌か、抜いた北へのロン。 */
    RON(false, null),
    /** 加槓した牌へのロン（槍槓）。 */
    CHANKAN(false, ScoringYaku.CHANKAN),
    /** 山が尽きた後の最後の打牌へのロン。 */
    HOUTEI(false, ScoringYaku.HOUTEI);

    private final boolean tsumo;
    private final ScoringYaku yaku;

    Source(boolean tsumo, ScoringYaku yaku) {
      this.tsumo = tsumo;
      this.yaku = yaku;
    }

    public boolean tsumo() {
      return tsumo;
    }

    /** この出どころで付く役。無ければ null。 */
    public ScoringYaku yaku() {
      return yaku;
    }
  }

  /** 和了牌がまだ無いことを表す値。 */
  private static final int NO_KIND = -1;

  private final Variant variant;

  /** このルールで赤 5 がある種類の集合。 */
  private final long redKinds;

  private final HandShape hand = new HandShape();
  private final Meld[] melds = new Meld[Variant.MAX_MELDS];
  private final int[] indicators = new int[MAX_INDICATORS];
  private final int[] uraIndicators = new int[MAX_INDICATORS];
  private int meldCount;
  private int indicatorCount;
  private int uraIndicatorCount;
  private int winningKind;
  private Source source;
  private long redFives;
  private int nuki;
  private int seatWind;
  private int roundWind;
  private RiichiState riichi;
  private boolean ippatsu;
  private boolean firstDraw;

  /** 作った直後は、{@link #clear} の後と同じ状態。 */
  public WinContext(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    redKinds = kindsWithRed(variant.tiles());
    clear();
  }

  public Variant variant() {
    return variant;
  }

  /** 手・面子・赤ドラ・抜きドラ・表示牌を空にし、和了牌を消す。風は東場の東家に、立直は無しに戻し、第一ツモでない状態にする。 */
  public void clear() {
    hand.clear();
    meldCount = 0;
    indicatorCount = 0;
    uraIndicatorCount = 0;
    winningKind = NO_KIND;
    source = null;
    redFives = 0;
    nuki = 0;
    seatWind = Tiles.TON;
    roundWind = Tiles.TON;
    riichi = RiichiState.NONE;
    ippatsu = false;
    firstDraw = false;
  }

  /** 純手牌と面子の数。呼び出し元が書き込む。エンジンの手は {@code copyFrom} でコピーできる。 */
  public HandShape hand() {
    return hand;
  }

  /**
   * 面子（副露と暗槓）を足す。{@link #hand} の面子の数は変えない。ルールに無い種類の面子や、面子にできない牌を含む面子なら
   * IllegalArgumentException を投げる。面子が既に上限まであれば IllegalStateException を投げる。
   */
  public void addMeld(Meld meld) {
    Objects.requireNonNull(meld, "meld");
    if (meldCount == melds.length) {
      throw new IllegalStateException("A hand holds at most " + melds.length + " melds");
    }
    if (!meld.type().in(variant) || (meld.kinds() & ~variant.meldableKinds()) != 0) {
      throw new IllegalArgumentException(variant + " has no meld " + meld);
    }
    melds[meldCount++] = meld;
  }

  public int meldCount() {
    return meldCount;
  }

  public Meld meld(int index) {
    return melds[Objects.checkIndex(index, meldCount)];
  }

  public void setWin(int kind, Source source) {
    requireKind(kind);
    this.source = Objects.requireNonNull(source, "source");
    winningKind = kind;
  }

  /** 和了牌の種類。{@link #setWin} の前は −1。 */
  public int winningKind() {
    return winningKind;
  }

  /** 和了牌の出どころ。{@link #setWin} の前は null。 */
  public Source source() {
    return source;
  }

  /** 純手牌（和了牌を含む）にある赤 5 を、種類の集合で書く。面子の赤 5 は {@link Meld} の情報から数える。 */
  public void setRedFives(long kinds) {
    if ((kinds & ~redKinds) != 0) {
      throw new IllegalArgumentException(variant + " has no red copy of kinds " + Long.toBinaryString(kinds));
    }
    redFives = kinds;
  }

  public long redFives() {
    return redFives;
  }

  /** 抜いた北の数を書く。北を抜かないルールでは 0 しか書けない。 */
  public void setNuki(int count) {
    int max = variant.has(RuleFlag.NUKI) ? Tiles.COPIES : 0;
    if (count < 0 || count > max) {
      throw new IllegalArgumentException(variant + " cannot hold " + count + " nuki");
    }
    nuki = count;
  }

  public int nuki() {
    return nuki;
  }

  /**
   * 自風と場風を書く。値の表し方は {@code Variant.seatWind}・{@code Variant.roundWind} と同じ。範囲の外なら
   * IllegalArgumentException を投げる。
   *
   * @param seatWind 自風の種類。東から数えて席の数だけある
   * @param roundWind 場風の種類。東から、そのルールで最後の局の場風まで
   */
  public void setWinds(int seatWind, int roundWind) {
    int lastRoundWind = variant.roundWind(variant.finalRound());
    if (seatWind < Tiles.TON
        || seatWind >= Tiles.TON + variant.seats()
        || roundWind < Tiles.TON
        || roundWind > lastRoundWind) {
      throw new IllegalArgumentException(
          "No seat wind " + seatWind + " with round wind " + roundWind + " in " + variant);
    }
    this.seatWind = seatWind;
    this.roundWind = roundWind;
  }

  public int seatWind() {
    return seatWind;
  }

  public int roundWind() {
    return roundWind;
  }

  /**
   * 立直の状態と一発を書く。一発は立直した手にだけ付き、立直していないのに一発を付けると IllegalArgumentException を投げる。
   * 嶺上牌のツモに一発は付かない（その前の自分の暗槓か抜きで一発が消える）。
   */
  public void setRiichi(RiichiState state, boolean ippatsu) {
    Objects.requireNonNull(state, "state");
    if (ippatsu && state == RiichiState.NONE) {
      throw new IllegalArgumentException("Ippatsu needs a riichi");
    }
    riichi = state;
    this.ippatsu = ippatsu;
  }

  public RiichiState riichi() {
    return riichi;
  }

  public boolean ippatsu() {
    return ippatsu;
  }

  /**
   * 和了牌が、誰も鳴いていない第一巡で引いた自分の最初のツモか（天和・地和）を書く。true にできるのは、面子・立直・抜きの無い手の
   * ツモだけ。
   */
  public void setFirstDraw(boolean firstDraw) {
    this.firstDraw = firstDraw;
  }

  public boolean firstDraw() {
    return firstDraw;
  }

  /**
   * 和了の時点でめくれているドラ表示牌の種類を、めくった順に足す。大明槓・加槓の槓ドラは後めくりなので、まだめくれていなければ
   * 足さない（嶺上開花の和了の時点では、まだめくれていない）。
   */
  public void addIndicator(int kind) {
    requireKind(kind);
    if (indicatorCount == indicators.length) {
      throw new IllegalStateException("At most " + indicators.length + " dora indicators");
    }
    indicators[indicatorCount++] = kind;
  }

  public int indicatorCount() {
    return indicatorCount;
  }

  public int indicator(int index) {
    return indicators[Objects.checkIndex(index, indicatorCount)];
  }

  /**
   * 裏ドラ表示牌の種類を、ドラ表示牌と同じ順に足す。裏ドラ表示牌は 0 枚（裏ドラを数えない）か、ドラ表示牌と同じ数にする。裏ドラを
   * 数えるのは立直した和了だけ。
   */
  public void addUraIndicator(int kind) {
    requireKind(kind);
    if (uraIndicatorCount == uraIndicators.length) {
      throw new IllegalStateException("At most " + uraIndicators.length + " ura indicators");
    }
    uraIndicators[uraIndicatorCount++] = kind;
  }

  public int uraIndicatorCount() {
    return uraIndicatorCount;
  }

  public int uraIndicator(int index) {
    return uraIndicators[Objects.checkIndex(index, uraIndicatorCount)];
  }

  private void requireKind(int kind) {
    if (kind < 0 || kind >= Tiles.KINDS || variant.tiles().copies(kind) == 0) {
      throw new IllegalArgumentException(variant + " has no tile kind " + kind);
    }
  }

  private static long kindsWithRed(TileSet tiles) {
    long kinds = 0;
    for (long fives = Tiles.FIVES & tiles.kinds(); fives != 0; fives &= fives - 1) {
      int kind = Long.numberOfTrailingZeros(fives);
      if (tiles.red(Tiles.physicalId(kind, Tiles.RED_COPY))) {
        kinds |= Tiles.bit(kind);
      }
    }
    return kinds;
  }

  @Override
  public String toString() {
    StringBuilder text = new StringBuilder("WinContext[").append(variant).append(' ').append(hand);
    for (int i = 0; i < meldCount; i++) {
      text.append(' ').append(melds[i]);
    }
    if (winningKind != NO_KIND) {
      text.append(" win=").append(Face.of(winningKind, false).mjai()).append(' ').append(source);
    }
    text.append(" seat=").append(Face.of(seatWind, false).mjai())
        .append(" round=").append(Face.of(roundWind, false).mjai())
        .append(' ').append(riichi);
    if (ippatsu) {
      text.append(" ippatsu");
    }
    if (firstDraw) {
      text.append(" firstDraw");
    }
    appendKinds(text, " dora=", indicators, indicatorCount);
    appendKinds(text, " ura=", uraIndicators, uraIndicatorCount);
    if (redFives != 0) {
      text.append(" red=");
      for (long rest = redFives; rest != 0; rest &= rest - 1) {
        text.append(Face.of(Long.numberOfTrailingZeros(rest), true).mjai());
      }
    }
    if (nuki > 0) {
      text.append(" nuki=").append(nuki);
    }
    return text.append(']').toString();
  }

  private static void appendKinds(StringBuilder text, String label, int[] kinds, int count) {
    if (count == 0) {
      return;
    }
    text.append(label);
    for (int i = 0; i < count; i++) {
      text.append(Face.of(kinds[i], false).mjai());
    }
  }
}
