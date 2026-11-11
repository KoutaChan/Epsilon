package com.epsilon.agent.facts;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 席ごとの公開情報を、記録 O だけから求める。求めるのは、今の面子、河にある種類、立直の後に通った種類、現物、危険度、点の順位。
 * 席は観測席から見た相対席で数える。判断ごとに {@link #load} で作り直し、そのときメモリは確保しない。同時に使えるのは 1 つの
 * スレッドだけ。
 *
 * <p>河にある種類と立直の後に通った種類は、{@code SeatView.riverKinds}・{@code riichiSafe} と同じ数え方をする。河には鳴かれた打牌も
 * 入る。立直の後に通った種類は、その席の立直の宣言牌より後に打たれた全ての牌（立直した席自身の打牌を含む）。どちらもその席がロン
 * できない種類（河による振聴と、立直の後の見逃しによる振聴）で、2 つを合わせたものがその席の現物（{@link #safe}）になる。
 *
 * <p>危険度（{@link #danger}。小さいほど安全）は、相手の現物 G と、字牌なら自分から見えている枚数で決まる。値は次の表のとおり。
 *
 * <table>
 *   <caption>危険度</caption>
 *   <tr><td>0</td><td>G にある</td></tr>
 *   <tr><td>1</td><td>字牌で、自分から見えている枚数（自分の純手牌を含む）が 3 以上。または 1・9 の筋（1 は 4、9 は 6 が G にある）</td></tr>
 *   <tr><td>2</td><td>字牌で見えている枚数が 2。または 2・3・7・8 の筋（2 は 5、3 は 6、7 は 4、8 は 5 が G にある）。または筋でない 1・9</td></tr>
 *   <tr><td>3</td><td>4・5・6 の両筋（4 は 1 と 7、5 は 2 と 8、6 は 3 と 9 が G にある）。または字牌で見えている枚数が 1</td></tr>
 *   <tr><td>4</td><td>字牌で見えている枚数が 0。または筋でない 2・3・7・8。または片筋の 4・5・6</td></tr>
 *   <tr><td>5</td><td>筋の無い 4・5・6</td></tr>
 * </table>
 */
public final class SeatFacts {

  /** 危険度の段階の数。 */
  public static final int DANGER_CLASSES = 6;

  /** 筋の間隔（数牌の数の差）。 */
  private static final int SUJI = 3;

  private static final int NONE = -1;

  private final Variant variant;
  private final int seats;
  private final Observation.EventRecord record = new Observation.EventRecord();

  /** 席ごとに、成立した順に並べた面子。添字は席 × 面子の上限 ＋ 何番目か。加槓したポンは、加槓の面子に置き換える。 */
  private final Meld[] melds;

  private final int[] meldCount;
  private final int[] openMelds;
  private final int[] kans;
  private final int[] riichiDeclaration;
  private final long[] riverKinds;
  private final long[] riichiSafe;
  private final int[] rank;
  private final int[] absoluteScores;
  private final int[] absoluteRanks;

  /** イベントを読み進める間、そこまでに立直を宣言した席。宣言した席のビットを立てる。 */
  private int declared;

  public SeatFacts(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    melds = new Meld[seats * Variant.MAX_MELDS];
    meldCount = new int[seats];
    openMelds = new int[seats];
    kans = new int[seats];
    riichiDeclaration = new int[seats];
    riverKinds = new long[seats];
    riichiSafe = new long[seats];
    rank = new int[seats];
    absoluteScores = new int[seats];
    absoluteRanks = new int[seats];
  }

  /**
   * 記録 O から、判断の時点での席ごとの公開情報を作る。記録のルール（四人打ちか三人打ちか）が作ったときと違えば
   * IllegalArgumentException を投げる。加槓の元になるポンが無いときや、面子が上限を超えるときは IllegalStateException を投げる。
   */
  public void load(Observation o) {
    begin(o);
    for (int event = 0; event < o.events(); event++) {
      add(o.readEvent(event, record));
    }
    finish(o);
  }

  /**
   * ほかの集計と一緒にイベントを 1 回だけ読むときに、最初に呼ぶ。続けて記録 O の全てのイベントを前から {@link #add} に渡し、最後に
   * {@link #finish} を呼ぶと、{@link #load} と同じ値になる。
   *
   * @throws IllegalArgumentException 記録のルール（四人打ちか三人打ちか）が、作ったときと違うとき
   */
  public void begin(Observation o) {
    if (o.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + o.variant());
    }
    Arrays.fill(meldCount, 0);
    Arrays.fill(openMelds, 0);
    Arrays.fill(kans, 0);
    Arrays.fill(riichiDeclaration, NONE);
    Arrays.fill(riverKinds, 0);
    Arrays.fill(riichiSafe, 0);
    declared = 0;
  }

  /**
   * {@link #begin} の後に、イベントを前から順に 1 つずつ加える。
   *
   * @throws IllegalStateException 加槓の元になるポンが無いとき、または面子が上限を超えるとき
   */
  public void add(Observation.EventRecord event) {
    int author = event.author();
    if (event.kind() == EventKind.DISCARD) {
      long bit = Tiles.bit(event.discardTile());
      for (int mask = declared; mask != 0; mask &= mask - 1) {
        riichiSafe[Integer.numberOfTrailingZeros(mask)] |= bit;
      }
      riverKinds[author] |= bit;
      if (event.discardRiichi()) {
        riichiDeclaration[author] = event.event();
        declared |= 1 << author;
      }
    } else if (event.kind() == EventKind.MELD) {
      addMeld(author, event.meld());
    }
  }

  /** 全てのイベントを {@link #add} に渡した後に呼び、点の順位を付けて読み終える。 */
  public void finish(Observation o) {
    for (int seat = 0; seat < seats; seat++) {
      absoluteScores[o.absoluteSeat(seat)] = o.score(seat);
    }
    Placement.ranks(absoluteScores, absoluteRanks);
    for (int seat = 0; seat < seats; seat++) {
      rank[seat] = absoluteRanks[o.absoluteSeat(seat)];
    }
  }

  /** 今の面子の数（副露と暗槓。加槓はポンを置き換える）。 */
  public int meldCount(int seat) {
    return meldCount[seat];
  }

  /** 今の面子のうち、成立した順で数えて指定した番号の面子を返す（加槓したポンは加槓の面子になっている）。 */
  public Meld meld(int seat, int index) {
    return melds[seat * Variant.MAX_MELDS + Objects.checkIndex(index, meldCount[seat])];
  }

  /** 門前を崩す面子（暗槓以外の面子）の数。0 なら門前。 */
  public int openMelds(int seat) {
    return openMelds[seat];
  }

  /** 槓子（大明槓・暗槓・加槓）の数。 */
  public int kans(int seat) {
    return kans[seat];
  }

  /** 立直の宣言牌を打ったイベントの番号（宣言していなければ −1）。宣言牌が通るまでは、立直はまだ成立していない。 */
  public int riichiDeclaration(int seat) {
    return riichiDeclaration[seat];
  }

  /** 河にある種類（鳴かれた打牌を含む）。 */
  public long riverKinds(int seat) {
    return riverKinds[seat];
  }

  /** その席の立直の宣言牌より後に、誰かが打った種類。 */
  public long riichiSafe(int seat) {
    return riichiSafe[seat];
  }

  /** 現物。その席がロンできない種類（河と、立直の後に通った種類）。 */
  public long safe(int seat) {
    return riverKinds[seat] | riichiSafe[seat];
  }

  /** 点の順位（0 始まり。同点は起家に近い席が上）。 */
  public int rank(int seat) {
    return rank[seat];
  }

  /**
   * 相手の席に対する、牌の種類の危険度を返す。危険度は相手の現物と、字牌なら見えている枚数で決まる。
   *
   * @param seat 相手の相対席
   * @param kind 牌の種類
   * @param tiles 種類ごとの枚数。ここから字牌の見えている枚数を読む
   * @return 危険度（0〜5。クラスの説明の表のとおり）
   */
  public int danger(int seat, int kind, TileCounts tiles) {
    return danger(safe(seat), kind, tiles.visible(kind) + tiles.hand(kind));
  }

  /**
   * 現物から、牌の種類の危険度を求める。数牌の数字は、色の中で 0 から数えた番号（1 が 0、9 が 8）で見る。
   *
   * @param safe 相手の現物
   * @param kind 牌の種類
   * @param seen 字牌なら、自分から見えている枚数（自分の純手牌を含む）
   * @return 危険度（0〜5）
   */
  static int danger(long safe, int kind, int seen) {
    if (Tiles.contains(safe, kind)) {
      return 0;
    }
    if (!Tiles.Group.of(kind).suited()) {
      return seen >= 3 ? 1 : seen == 2 ? 2 : seen == 1 ? 3 : 4;
    }
    return switch (Tiles.number(kind)) {
      case 0 -> Tiles.contains(safe, kind + SUJI) ? 1 : 2;
      case 8 -> Tiles.contains(safe, kind - SUJI) ? 1 : 2;
      case 1, 2 -> Tiles.contains(safe, kind + SUJI) ? 2 : 4;
      case 6, 7 -> Tiles.contains(safe, kind - SUJI) ? 2 : 4;
      default -> {
        boolean low = Tiles.contains(safe, kind - SUJI);
        boolean high = Tiles.contains(safe, kind + SUJI);
        yield low && high ? 3 : low || high ? 4 : 5;
      }
    };
  }

  private void addMeld(int seat, Meld meld) {
    int first = seat * Variant.MAX_MELDS;
    if (meld.type() == Meld.Type.KAKAN) {
      for (int index = first; index < first + meldCount[seat]; index++) {
        if (melds[index].type() == Meld.Type.PON && melds[index].base() == meld.base()) {
          melds[index] = meld;
          kans[seat]++;
          return;
        }
      }
      throw new IllegalStateException("Seat " + seat + " adds " + meld + " without its pon");
    }
    if (meldCount[seat] == Variant.MAX_MELDS) {
      throw new IllegalStateException("Seat " + seat + " holds more than " + Variant.MAX_MELDS + " melds");
    }
    melds[first + meldCount[seat]++] = meld;
    if (!meld.type().concealed()) {
      openMelds[seat]++;
    }
    if (meld.type().kan()) {
      kans[seat]++;
    }
  }
}
