package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 天鳳の牌譜で N タグの m に書く、面子のコードを読み書きする。このレコードは、面子とその物理牌を持つ。
 *
 * <p>コードは 16 bit で、フィールドは面子の種類ごとの enum の宣言の順に、下位の bit から並ぶ。鳴き元はどの種類でも最下位のフィールドに
 * ある。その値は d = (元 − 席) mod 4 で、天鳳の卓の 4 席（三人打ちは空き席を含む）の上で、鳴いた席から鳴き元の席までの距離を表す
 * （暗槓は 0）。三人打ちでは、同じ d でも鳴いた席によって上家か下家かが変わるので、読み書きには鳴いた席（タグの席）が要る。
 *
 * <p>鳴き元の次に並ぶフラグの bit を、チー → ポン → 加槓 → 北抜きの順に見て種類を決める。フラグが立っていないコードは暗槓か大明槓。
 * 北抜きは面子でないのでこの型には読まず、北抜きのコードを見分けて読み書きする別のメソッドで扱う。
 *
 * @param meld 面子
 * @param called 鳴いた牌。暗槓は −1、加槓は元のポンで鳴いた牌
 * @param hand 手から出した牌（昇順）。チー・ポンは 2 枚、大明槓は 3 枚、暗槓は 4 枚、加槓は足した 1 枚
 */
public record TenhouMeldCode(Meld meld, int called, int[] hand) {

  /** コードの bit 数。 */
  private static final int BITS = 16;

  /** 順子の枚数。チーで鳴いた牌が順子の何枚目かの通り数でもある。 */
  private static final int RUN = Meld.Type.CHI.tiles();

  /** 1 色の中で、順子の最初の牌になれる数の通り数（1〜7 の 7 通り）。 */
  private static final int RUN_STARTS = Tiles.SUIT_SIZE - RUN + 1;

  private static final int PON_TILES = Meld.Type.PON.tiles();

  /** 数牌の色（萬・筒・索の順）。この並びでの位置が、チーの RUN での色の番号になる。 */
  private static final Tiles.Group[] SUITS =
      Arrays.stream(Tiles.Group.values()).filter(Tiles.Group::suited).toArray(Tiles.Group[]::new);

  /** コードのフィールド。幅を bit 数で持つ。 */
  private interface Field {
    int width();
  }

  /** チーのフィールド。LOW・MIDDLE・HIGH は、順子の各牌がその種類の何枚目か。RUN は (色 × 7 ＋ 始まりの数 − 1) × 3 ＋ 鳴いた位置。 */
  private enum Chi implements Field {
    FROM(2),
    CHI(1),
    LOW(2),
    MIDDLE(2),
    HIGH(2),
    PAD(1),
    RUN(6);

    private final int width;

    Chi(int width) {
      this.width = width;
    }

    @Override
    public int width() {
      return width;
    }
  }

  /**
   * ポンと加槓のフィールド。OTHER は、ポンに入らない 1 枚（加槓なら足した牌）が何枚目か。CALLED は、種類 × 3 ＋ 鳴いた牌の位置。
   * 位置は、ポンの 3 枚を何枚目かの順に並べた中で数える。
   */
  private enum Triplet implements Field {
    FROM(2),
    CHI(1),
    PON(1),
    KAKAN(1),
    OTHER(2),
    PAD(2),
    CALLED(7);

    private final int width;

    Triplet(int width) {
      this.width = width;
    }

    @Override
    public int width() {
      return width;
    }
  }

  /** 暗槓と大明槓のフィールド。TILE は、大明槓なら鳴いた牌、暗槓ならその種類で番号の最も小さい物理 id。 */
  private enum Kan implements Field {
    FROM(2),
    MARKS(6),
    TILE(8);

    private final int width;

    Kan(int width) {
      this.width = width;
    }

    @Override
    public int width() {
      return width;
    }
  }

  /** 北抜きのフィールド。鳴き元は 0 で、フラグは北抜きの bit だけが立つ。TILE は抜いた北の物理 id。 */
  private enum Nuki implements Field {
    FROM(2),
    CHI(1),
    PON(1),
    KAKAN(1),
    NUKI(1),
    PAD(2),
    TILE(8);

    private final int width;

    Nuki(int width) {
      this.width = width;
    }

    @Override
    public int width() {
      return width;
    }
  }

  /** 各フィールドが何 bit 目から始まるか。宣言の順に幅を足し上げて、一度だけ求める。幅の合計は 16 bit でなければならない。 */
  private static final class Layout<F extends Enum<F> & Field> {

    private final int[] offsets;

    Layout(Class<F> type) {
      F[] fields = type.getEnumConstants();
      offsets = new int[fields.length];
      int offset = 0;
      for (F field : fields) {
        offsets[field.ordinal()] = offset;
        offset += field.width();
      }
      if (offset != BITS) {
        throw new IllegalStateException(type.getSimpleName() + " covers " + offset + " bits, not " + BITS);
      }
    }

    int get(int code, F field) {
      return code >>> offsets[field.ordinal()] & (1 << field.width()) - 1;
    }

    int put(F field, int value) {
      if (value < 0 || value >= 1 << field.width()) {
        throw new IllegalArgumentException(field + " cannot hold " + value);
      }
      return value << offsets[field.ordinal()];
    }

    /** そのフィールドより下位にある、全てのフィールドの bit のマスク。 */
    int below(F field) {
      return (1 << offsets[field.ordinal()]) - 1;
    }
  }

  private static final Layout<Chi> CHI = new Layout<>(Chi.class);
  private static final Layout<Triplet> TRIPLET = new Layout<>(Triplet.class);
  private static final Layout<Kan> KAN = new Layout<>(Kan.class);
  private static final Layout<Nuki> NUKI = new Layout<>(Nuki.class);

  /** 北抜きのコードで、鳴き元とフラグのフィールドが取る値。 */
  private static final int NUKI_MARKS = NUKI.put(Nuki.NUKI, 1);

  public TenhouMeldCode {
    Objects.requireNonNull(meld, "meld");
    hand = hand.clone();
    Arrays.sort(hand);
  }

  /**
   * 面子のコードを読む。
   *
   * @param code N タグの m
   * @param variant 卓のルール
   * @param seat 鳴いた席（タグの席）
   * @return 面子とその物理牌
   * @throws IllegalArgumentException 面子のコードでない値か、北抜きのコードか、チーの無いルールでチーのコードを渡したとき
   */
  public static TenhouMeldCode decode(int code, Variant variant, int seat) {
    Objects.checkIndex(seat, variant.seats());
    if (code < 0 || code >= 1 << BITS) {
      throw new IllegalArgumentException("Tenhou meld codes have " + BITS + " bits: " + code);
    }
    if (TRIPLET.get(code, Triplet.CHI) != 0) {
      return chi(code, variant, seat);
    }
    if (TRIPLET.get(code, Triplet.PON) != 0) {
      return triplet(code, variant, seat, false);
    }
    if (TRIPLET.get(code, Triplet.KAKAN) != 0) {
      return triplet(code, variant, seat, true);
    }
    if (nuki(code)) {
      throw new IllegalArgumentException("Tenhou code " + code + " is a nuki, not a meld");
    }
    return kan(code, variant, seat);
  }

  /**
   * 北抜きのコードか。北抜きのコードでは、鳴き元とフラグのフィールド（下位 6 bit）で北抜きのフラグだけが立つ。
   *
   * @param code N タグの m
   */
  public static boolean nuki(int code) {
    return code >= 0 && code < 1 << BITS && (code & NUKI.below(Nuki.PAD)) == NUKI_MARKS;
  }

  /**
   * 北抜きのコードから、抜いた北を読む。
   *
   * @param code N タグの m
   * @param variant 卓のルール
   * @return 抜いた北の物理 id
   * @throws IllegalArgumentException 北抜きの無いルールのときか、北抜きのコードでない値のとき
   */
  public static int nukiTile(int code, Variant variant) {
    if (!variant.has(RuleFlag.NUKI)) {
      throw new IllegalArgumentException(variant + " has no nuki, but Tenhou code " + code + " is one");
    }
    int tile = NUKI.get(code, Nuki.TILE);
    if (!nuki(code) || NUKI.get(code, Nuki.PAD) != 0 || Tiles.kind(tile) != Tiles.PEI) {
      throw new IllegalArgumentException("Not a Tenhou nuki code: " + code);
    }
    return tile;
  }

  /**
   * 北抜きのコードを作る。
   *
   * @param tile 抜く北の物理 id
   * @throws IllegalArgumentException 北でない牌のとき
   */
  public static int encodeNuki(int tile) {
    if (tile < 0 || tile >= Tiles.PHYSICAL_IDS || Tiles.kind(tile) != Tiles.PEI) {
      throw new IllegalArgumentException("Only a north is taken out, not tile " + tile);
    }
    return NUKI_MARKS | NUKI.put(Nuki.TILE, tile);
  }

  /**
   * この面子のコードを作る。
   *
   * @param variant 卓のルール
   * @param seat 鳴いた席（タグの席）
   * @throws IllegalArgumentException 物理牌が面子と合わないとき
   */
  public int encode(Variant variant, int seat) {
    int code =
        switch (meld.type()) {
          case CHI -> encodeChi(variant, seat);
          case PON, KAKAN -> encodeTriplet(variant, seat);
          case DAIMINKAN -> KAN.put(Kan.FROM, distance(variant, seat)) | KAN.put(Kan.TILE, called);
          case ANKAN -> KAN.put(Kan.TILE, Tiles.physicalId(meld.base(), 0));
        };
    if (!decode(code, variant, seat).equals(this)) {
      throw new IllegalArgumentException("Tiles " + called + " and " + Arrays.toString(hand) + " do not make " + meld);
    }
    return code;
  }

  /** 手から出した牌（昇順）を、配列のコピーで返す。 */
  @Override
  public int[] hand() {
    return hand.clone();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof TenhouMeldCode code
        && meld == code.meld
        && called == code.called
        && Arrays.equals(hand, code.hand);
  }

  @Override
  public int hashCode() {
    return Objects.hash(meld, called, Arrays.hashCode(hand));
  }

  @Override
  public String toString() {
    return "TenhouMeldCode[" + meld + " called=" + called + " hand=" + Arrays.toString(hand) + "]";
  }

  private static TenhouMeldCode chi(int code, Variant variant, int seat) {
    if (!variant.has(RuleFlag.CHI)) {
      throw new IllegalArgumentException(variant + " has no chi, but Tenhou code " + code + " is one");
    }
    int run = CHI.get(code, Chi.RUN);
    int sequence = run / RUN;
    int position = run % RUN;
    if (source(CHI.get(code, Chi.FROM), variant, seat) != Meld.RelativeSource.KAMICHA
        || CHI.get(code, Chi.PAD) != 0
        || sequence >= SUITS.length * RUN_STARTS) {
      throw new IllegalArgumentException("Not a Tenhou chi code: " + code);
    }
    int base = SUITS[sequence / RUN_STARTS].first() + sequence % RUN_STARTS;
    int[] tiles = {
      Tiles.physicalId(base, CHI.get(code, Chi.LOW)),
      Tiles.physicalId(base + 1, CHI.get(code, Chi.MIDDLE)),
      Tiles.physicalId(base + 2, CHI.get(code, Chi.HIGH))
    };
    TileSet set = requireTiles(variant, tiles);
    int called = tiles[position];
    int[] hand = without(tiles, called);
    return new TenhouMeldCode(Meld.chi(base, base + position, aka(set, called, hand)), called, hand);
  }

  /**
   * ポンか加槓のコードを読む。
   *
   * @param added 加槓なら真
   */
  private static TenhouMeldCode triplet(int code, Variant variant, int seat, boolean added) {
    int calledField = TRIPLET.get(code, Triplet.CALLED);
    int kind = calledField / PON_TILES;
    if (TRIPLET.get(code, Triplet.PAD) != 0
        || !added && TRIPLET.get(code, Triplet.KAKAN) != 0
        || kind >= Tiles.KINDS) {
      throw new IllegalArgumentException("Not a Tenhou pon code: " + code);
    }
    int other = Tiles.physicalId(kind, TRIPLET.get(code, Triplet.OTHER));
    int[] pon = without(copies(kind), other);
    TileSet set = requireTiles(variant, copies(kind));
    int called = pon[calledField % PON_TILES];
    int[] ponHand = without(pon, called);
    Meld meld = Meld.pon(kind, source(TRIPLET.get(code, Triplet.FROM), variant, seat), aka(set, called, ponHand));
    if (!added) {
      return new TenhouMeldCode(meld, called, ponHand);
    }
    return new TenhouMeldCode(Meld.kakan(meld, set.red(other)), called, new int[] {other});
  }

  private static TenhouMeldCode kan(int code, Variant variant, int seat) {
    int tile = KAN.get(code, Kan.TILE);
    int kind = Tiles.kind(tile);
    if (KAN.get(code, Kan.MARKS) != 0 || kind >= Tiles.KINDS) {
      throw new IllegalArgumentException("Not a Tenhou kan code: " + code);
    }
    int[] all = copies(kind);
    TileSet set = requireTiles(variant, all);
    int from = KAN.get(code, Kan.FROM);
    if (from == 0) {
      return new TenhouMeldCode(Meld.ankan(kind, aka(set, -1, all) != Meld.AkaSource.NONE), -1, all);
    }
    int[] hand = without(all, tile);
    return new TenhouMeldCode(Meld.daiminkan(kind, source(from, variant, seat), aka(set, tile, hand)), tile, hand);
  }

  private int encodeChi(Variant variant, int seat) {
    int base = meld.base();
    int[] copies = new int[RUN];
    for (int tile : tilesWithCalled()) {
      int position = Tiles.kind(tile) - base;
      if (position < 0 || position >= RUN) {
        throw new IllegalArgumentException("Tile " + tile + " is not in " + meld);
      }
      copies[position] = Tiles.copy(tile);
    }
    int suit = Arrays.asList(SUITS).indexOf(Tiles.Group.of(base));
    int run = (suit * RUN_STARTS + Tiles.number(base)) * RUN + meld.called() - base;
    return CHI.put(Chi.FROM, distance(variant, seat))
        | CHI.put(Chi.CHI, 1)
        | CHI.put(Chi.LOW, copies[0])
        | CHI.put(Chi.MIDDLE, copies[1])
        | CHI.put(Chi.HIGH, copies[2])
        | CHI.put(Chi.RUN, run);
  }

  private int encodeTriplet(Variant variant, int seat) {
    int kind = meld.base();
    boolean added = meld.type() == Meld.Type.KAKAN;
    int other;
    int[] pon;
    if (added) {
      other = hand[0];
      pon = without(copies(kind), other);
    } else {
      pon = tilesWithCalled();
      other = -1;
      for (int tile : copies(kind)) {
        if (Arrays.binarySearch(pon, tile) < 0) {
          other = tile;
        }
      }
    }
    int position = Arrays.binarySearch(pon, called);
    if (position < 0) {
      throw new IllegalArgumentException("Tile " + called + " is not in " + meld);
    }
    return TRIPLET.put(Triplet.FROM, distance(variant, seat))
        | TRIPLET.put(added ? Triplet.KAKAN : Triplet.PON, 1)
        | TRIPLET.put(Triplet.OTHER, Tiles.copy(other))
        | TRIPLET.put(Triplet.CALLED, kind * PON_TILES + position);
  }

  /** 鳴いた牌と手から出した牌を合わせ、昇順に並べて返す。 */
  private int[] tilesWithCalled() {
    int[] tiles = Arrays.copyOf(hand, hand.length + 1);
    tiles[hand.length] = called;
    Arrays.sort(tiles);
    return tiles;
  }

  /** 鳴き元の d を求める。天鳳の卓の 4 席の上で数えた、鳴いた席から鳴き元の席までの距離。 */
  private int distance(Variant variant, int seat) {
    return Math.floorMod(meld.source().seat(seat, variant.seats()) - seat, TenhouRound.TAG_SEATS);
  }

  /** 鳴いた席から見た、鳴き元の席の向きを返す。d が自分か空き席を指せば IllegalArgumentException を投げる。 */
  private static Meld.RelativeSource source(int distance, Variant variant, int seat) {
    int source = (seat + distance) % TenhouRound.TAG_SEATS;
    if (distance == 0 || source >= variant.seats()) {
      throw new IllegalArgumentException("Seat " + seat + " cannot call from distance " + distance);
    }
    return Meld.RelativeSource.of(seat, source, variant.seats());
  }

  /**
   * 面子の赤がどこから来たか。鳴いた牌が赤なら {@code CALLED_TILE}、手から出した牌に赤があれば {@code CONSUMED_HAND_TILE}、ほかは
   * {@code NONE}。
   *
   * @param called 鳴いた牌。無ければ −1
   */
  private static Meld.AkaSource aka(TileSet set, int called, int[] hand) {
    if (called >= 0 && set.red(called)) {
      return Meld.AkaSource.CALLED_TILE;
    }
    for (int tile : hand) {
      if (set.red(tile)) {
        return Meld.AkaSource.CONSUMED_HAND_TILE;
      }
    }
    return Meld.AkaSource.NONE;
  }

  private static TileSet requireTiles(Variant variant, int[] tiles) {
    TileSet set = variant.tiles();
    for (int tile : tiles) {
      if (!set.contains(tile)) {
        throw new IllegalArgumentException("Tile " + tile + " is not in " + set);
      }
    }
    return set;
  }

  /** その種類の 4 枚の物理 id（番号の小さい順）。 */
  private static int[] copies(int kind) {
    int[] copies = new int[Tiles.COPIES];
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      copies[copy] = Tiles.physicalId(kind, copy);
    }
    return copies;
  }

  /** 昇順の列から牌を 1 枚除き、残りを昇順のまま返す。その牌が無ければ IllegalArgumentException を投げる。 */
  private static int[] without(int[] tiles, int tile) {
    int[] rest = new int[tiles.length - 1];
    int size = 0;
    for (int each : tiles) {
      if (each != tile) {
        rest[size++] = each;
      }
    }
    if (size != rest.length) {
      throw new IllegalArgumentException("Tile " + tile + " is not in " + Arrays.toString(tiles));
    }
    return rest;
  }
}
