package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 局の公開イベントの記録。追記するだけで、イベントの番号は局の中で 0 から振る通し番号。フィールドは基本型の配列で持つ。
 * 書き換えるのは、鳴きの面子を足すときに、元の打牌へ鳴いた席を書くことだけ。容量は {@link Variant#maxPublicEvents} で、
 * 超えると IllegalStateException を投げる。
 *
 * <p>どのフィールドを持つかは種類で違う。牌の種類と赤は打牌と抜きが、ツモ切り・立直・鳴いた席は打牌が持つ。面子は鳴き・暗槓・
 * 加槓（加槓した後の面子）が持つ。持たないフィールドは {@link #NONE} か false。
 */
public final class EventLog {

  /** フィールドに値が無いことを表す。鳴かれていない打牌の、鳴いた席など。 */
  public static final int NONE = -1;

  private static final EventKind[] KINDS = EventKind.values();

  private final Variant variant;

  /** 1 席の河に並ぶ打牌の上限。打牌するのはツモ（山の枚数以下）の後か、チー・ポン（1 席 4 回以下）の後だけなので、その和。 */
  private final int riverCapacity;

  private final byte[] kind;
  private final byte[] seat;
  private final byte[] tile;
  private final byte[] calledBy;
  private final short[] meld;
  private final boolean[] red;
  private final boolean[] tsumogiri;
  private final boolean[] riichi;

  /** 席ごとの河。席 × 河の上限 ＋ i の位置に、その席が i 番目に打った牌のイベントの番号を置く。鳴かれた打牌も河に残す。 */
  private final byte[] river;

  private final byte[] riverSize;
  private int size;
  private int dahai;
  private int interruptions;

  EventLog(Variant variant) {
    this.variant = variant;
    riverCapacity = variant.liveWall() + Variant.MAX_MELDS;
    int capacity = variant.maxPublicEvents();
    kind = new byte[capacity];
    seat = new byte[capacity];
    tile = new byte[capacity];
    calledBy = new byte[capacity];
    meld = new short[capacity];
    red = new boolean[capacity];
    tsumogiri = new boolean[capacity];
    riichi = new boolean[capacity];
    river = new byte[variant.seats() * riverCapacity];
    riverSize = new byte[variant.seats()];
  }

  /** イベントの数。次に足すイベントの番号でもある。 */
  public int size() {
    return size;
  }

  public EventKind kind(int event) {
    return KINDS[kind[at(event)]];
  }

  /** イベントを起こした絶対席。 */
  public int seat(int event) {
    return seat[at(event)];
  }

  /** 打牌した牌、または抜いた牌の種類。 */
  public int tile(int event) {
    return tile[at(event)];
  }

  public boolean red(int event) {
    return red[at(event)];
  }

  public boolean tsumogiri(int event) {
    return tsumogiri[at(event)];
  }

  /** 打牌が立直の宣言牌なら true。 */
  public boolean riichi(int event) {
    return riichi[at(event)];
  }

  /** 打牌を鳴いた絶対席。 */
  public int calledBy(int event) {
    return calledBy[at(event)];
  }

  /** 面子の正準の番号（{@link Meld#id}）。ポンのイベントは、後で加槓してもポンの番号のまま。 */
  public int meld(int event) {
    return meld[at(event)];
  }

  /** 全ての席を合わせた打牌の数。 */
  public int dahai() {
    return dahai;
  }

  /** 鳴き・暗槓・加槓・抜きの数。0 なら、局が始まってからまだ一度も巡が途切れていない。 */
  public int interruptions() {
    return interruptions;
  }

  public int riverSize(int seat) {
    return riverSize[seat];
  }

  /**
   * 河にある打牌のイベントの番号。
   *
   * @param seat 席
   * @param index 河の何番目か
   */
  public int river(int seat, int index) {
    return river[seat * riverCapacity + Objects.checkIndex(index, riverSize[seat])];
  }

  void clear() {
    size = 0;
    dahai = 0;
    interruptions = 0;
    Arrays.fill(riverSize, (byte) 0);
  }

  int appendDiscard(int author, int tileKind, boolean redFive, boolean fromDraw, boolean declaresRiichi) {
    if (tileKind < 0 || tileKind >= Tiles.KINDS || variant.tiles().copies(tileKind) == 0) {
      throw new IllegalArgumentException(variant + " has no tile kind " + tileKind);
    }
    int place = riverSize[Objects.checkIndex(author, riverSize.length)];
    if (place == riverCapacity) {
      throw new IllegalStateException("Seat " + author + " discards at most " + riverCapacity + " tiles");
    }
    int event = next(EventKind.DISCARD, author);
    tile[event] = (byte) tileKind;
    red[event] = redFive;
    tsumogiri[event] = fromDraw;
    riichi[event] = declaresRiichi;
    river[author * riverCapacity + place] = (byte) event;
    riverSize[author]++;
    dahai++;
    return event;
  }

  /**
   * チー・ポン・大明槓・暗槓を足す。鳴きなら、最後のイベントが鳴いた牌の打牌であることを確かめる。鳴かれた席の打牌で、まだ
   * 鳴かれておらず、牌の種類と赤が鳴いた牌と同じでなければならない。合っていればその打牌に鳴いた席を書き、合っていなければ記録を
   * 変えずに IllegalStateException を投げる。
   */
  int appendMeld(int author, int meldId) {
    Meld added = Meld.of(meldId);
    if (added.type() == Meld.Type.KAKAN || !added.type().in(variant)) {
      throw new IllegalArgumentException(variant + " cannot append " + added + " as a meld");
    }
    requireRoom();
    if (added.source() != Meld.RelativeSource.NONE) {
      markCalled(author, added);
    }
    int event = next(EventKind.MELD, author);
    meld[event] = (short) meldId;
    interruptions++;
    return event;
  }

  /** 加槓を足す。元のポンのイベントは書き換えない。 */
  int appendKakan(int author, int meldId) {
    Meld kakan = Meld.of(meldId);
    if (kakan.type() != Meld.Type.KAKAN) {
      throw new IllegalArgumentException("Not a kakan: " + kakan);
    }
    requireRoom();
    requirePon(author, kakan);
    int event = next(EventKind.KAKAN, author);
    meld[event] = (short) meldId;
    interruptions++;
    return event;
  }

  int appendNuki(int author) {
    if (!variant.has(RuleFlag.NUKI)) {
      throw new IllegalStateException(variant + " has no nuki");
    }
    int event = next(EventKind.NUKI, author);
    tile[event] = Tiles.PEI;
    interruptions++;
    return event;
  }

  void copyFrom(EventLog other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    System.arraycopy(other.kind, 0, kind, 0, kind.length);
    System.arraycopy(other.seat, 0, seat, 0, seat.length);
    System.arraycopy(other.tile, 0, tile, 0, tile.length);
    System.arraycopy(other.calledBy, 0, calledBy, 0, calledBy.length);
    System.arraycopy(other.meld, 0, meld, 0, meld.length);
    System.arraycopy(other.red, 0, red, 0, red.length);
    System.arraycopy(other.tsumogiri, 0, tsumogiri, 0, tsumogiri.length);
    System.arraycopy(other.riichi, 0, riichi, 0, riichi.length);
    System.arraycopy(other.river, 0, river, 0, river.length);
    System.arraycopy(other.riverSize, 0, riverSize, 0, riverSize.length);
    size = other.size;
    dahai = other.dahai;
    interruptions = other.interruptions;
  }

  /** 次のイベントの種類と起こした席を書き、ほかのフィールドを空にして、そのイベントの番号を返す。 */
  private int next(EventKind eventKind, int author) {
    requireRoom();
    int event = size;
    kind[event] = (byte) eventKind.ordinal();
    seat[event] = (byte) Objects.checkIndex(author, riverSize.length);
    tile[event] = NONE;
    calledBy[event] = NONE;
    meld[event] = NONE;
    red[event] = false;
    tsumogiri[event] = false;
    riichi[event] = false;
    size++;
    return event;
  }

  private void markCalled(int caller, Meld called) {
    int last = size - 1;
    int source = called.source().seat(caller, variant.seats());
    boolean calledRed = called.aka() == Meld.AkaSource.CALLED_TILE;
    if (last < 0
        || kind[last] != EventKind.DISCARD.ordinal()
        || calledBy[last] != NONE
        || seat[last] != source
        || tile[last] != called.called()
        || red[last] != calledRed) {
      throw new IllegalStateException(called + " by seat " + caller + " does not follow its called discard");
    }
    calledBy[last] = (byte) caller;
  }

  /**
   * その席に加槓の元になるポンがあることと、加槓がそのポンに 4 枚目を足した面子であることを確かめる。ポンが無ければ
   * IllegalStateException を、面子が合わなければ IllegalArgumentException を投げる。
   */
  private void requirePon(int author, Meld kakan) {
    for (int event = 0; event < size; event++) {
      if (kind[event] != EventKind.MELD.ordinal() || seat[event] != author) {
        continue;
      }
      Meld pon = Meld.of(meld[event]);
      if (pon.type() == Meld.Type.PON && pon.base() == kakan.base()) {
        if (Meld.kakan(pon, kakan.aka() == Meld.AkaSource.ADDED_KAN_TILE) != kakan) {
          throw new IllegalArgumentException(kakan + " is not an upgrade of " + pon);
        }
        return;
      }
    }
    throw new IllegalStateException("Seat " + author + " has no pon for " + kakan);
  }

  private void requireRoom() {
    if (size == kind.length) {
      throw new IllegalStateException("A round has at most " + kind.length + " public events");
    }
  }

  private int at(int event) {
    return Objects.checkIndex(event, size);
  }
}
