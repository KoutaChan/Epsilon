package com.epsilon.mahjong.action;

import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.VariantScoped;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.List;

/**
 * 面子。不変の値で、番号から表で引く。同じ面子には、いつも同じインスタンスを返す。
 *
 * <p>番号 = (((種別 × 34 ＋ 最小の種類) × 4 ＋ 鳴き元) × 3 ＋ チーの鳴いた位置) × 4 ＋ 赤の由来。種別・鳴き元・赤の由来は
 * ordinal で、チー以外は鳴いた位置を 0 とする。系譜の記録は面子の番号を持つので、この式と並び方は、記録を読む側との契約になる。
 * 三人打ちも同じ表を使う。
 */
public final class Meld {

  public enum Type implements VariantScoped {
    CHI(3, RuleFlag.CHI),
    PON(3),
    DAIMINKAN(4),
    ANKAN(4),
    KAKAN(4);

    private final int tiles;
    private final RuleFlag requires;

    Type(int tiles) {
      this(tiles, null);
    }

    Type(int tiles, RuleFlag requires) {
      this.tiles = tiles;
      this.requires = requires;
    }

    /** 面子の枚数。 */
    public int tiles() {
      return tiles;
    }

    /** 槓子（同じ種類の 4 枚を全て使う面子）なら true。 */
    public boolean kan() {
      return tiles == Tiles.COPIES;
    }

    /** 門前を崩さないなら true（暗槓だけ）。 */
    public boolean concealed() {
      return this == ANKAN;
    }

    @Override
    public RuleFlag requires() {
      return requires;
    }
  }

  /**
   * 鳴いた席から見た鳴き元。d = (鳴き元 − 鳴いた席) mod 席の数 とすると、d が 1 なら下家、席の数 − 1 なら上家、四人打ちで 2 なら
   * 対面。暗槓は鳴き元なし。
   */
  public enum RelativeSource {
    NONE,
    SHIMOCHA,
    TOIMEN,
    KAMICHA;

    public static RelativeSource of(int seat, int source, int seats) {
      requireSeats(seats);
      if (seat < 0 || seat >= seats || source < 0 || source >= seats) {
        throw new IllegalArgumentException("Seats " + seat + " and " + source + " must be below " + seats);
      }
      int distance = Math.floorMod(source - seat, seats);
      if (distance == 0) {
        return NONE;
      }
      if (distance == 1) {
        return SHIMOCHA;
      }
      return distance == seats - 1 ? KAMICHA : TOIMEN;
    }

    /**
     * 鳴き元の席番号を返す。鳴き元が無ければ面子の持ち主の席。
     *
     * @param owner 面子を持つ席
     * @param seats 席の数
     */
    public int seat(int owner, int seats) {
      requireSeats(seats);
      int distance =
          switch (this) {
            case NONE -> 0;
            case SHIMOCHA -> 1;
            case KAMICHA -> seats - 1;
            case TOIMEN -> {
              if (seats != 4) {
                throw new IllegalArgumentException("TOIMEN needs four seats, got " + seats);
              }
              yield 2;
            }
          };
      return (owner + distance) % seats;
    }

    private static void requireSeats(int seats) {
      if (seats != 3 && seats != 4) {
        throw new IllegalArgumentException("Seats must be 3 or 4, got " + seats);
      }
    }
  }

  public enum AkaSource {
    NONE,
    /** 鳴いた牌が赤。 */
    CALLED_TILE,
    /** 手から出して面子にした牌が赤（赤を含む暗槓もこれ）。 */
    CONSUMED_HAND_TILE,
    /** 加槓で足した牌が赤。 */
    ADDED_KAN_TILE
  }

  /** 順子の枚数。チーの鳴いた位置の数でもある。 */
  static final int RUN = 3;

  /** 1 色で順子の始まりになれる数字の数（1〜7 の 7 通り）。 */
  static final int RUN_STARTS = Tiles.SUIT_SIZE - RUN + 1;

  private static final Type[] TYPES = Type.values();
  private static final RelativeSource[] SOURCES = RelativeSource.values();
  private static final AkaSource[] AKA_SOURCES = AkaSource.values();
  private static final Type[] OPEN_TRIPLETS = {Type.PON, Type.DAIMINKAN, Type.KAKAN};
  private static final RelativeSource[] OPEN_SOURCES = {
    RelativeSource.SHIMOCHA, RelativeSource.TOIMEN, RelativeSource.KAMICHA
  };

  /** 番号の数（使わない番号を含む）。 */
  public static final int IDS = TYPES.length * Tiles.KINDS * SOURCES.length * RUN * AKA_SOURCES.length;

  private static final Meld[] TABLE = table();

  private final int id;
  private final Type type;
  private final int base;
  private final int called;
  private final RelativeSource source;
  private final AkaSource aka;

  private Meld(Type type, int base, int called, RelativeSource source, AkaSource aka) {
    this.type = type;
    this.base = base;
    this.called = called;
    this.source = source;
    this.aka = aka;
    id = idOf(type, base, called, source, aka);
  }

  /** 番号の面子を返す。使わない番号なら IllegalArgumentException を投げる。 */
  public static Meld of(int id) {
    Meld meld = id >= 0 && id < IDS ? TABLE[id] : null;
    if (meld == null) {
      throw new IllegalArgumentException("Not a meld id: " + id);
    }
    return meld;
  }

  /**
   * チーの面子を返す。鳴き元はいつも上家。
   *
   * @param base 順子の最小の種類
   * @param called 鳴いた牌の種類
   * @param aka 赤の由来
   */
  public static Meld chi(int base, int called, AkaSource aka) {
    return lookup(Type.CHI, base, called, RelativeSource.KAMICHA, aka);
  }

  public static Meld pon(int kind, RelativeSource source, AkaSource aka) {
    return lookup(Type.PON, kind, kind, source, aka);
  }

  public static Meld daiminkan(int kind, RelativeSource source, AkaSource aka) {
    return lookup(Type.DAIMINKAN, kind, kind, source, aka);
  }

  /**
   * 暗槓の面子を返す。
   *
   * @param kind 槓にする牌の種類
   * @param red 4 枚に赤の 5 を含むなら true
   */
  public static Meld ankan(int kind, boolean red) {
    return lookup(Type.ANKAN, kind, -1, RelativeSource.NONE, red ? AkaSource.CONSUMED_HAND_TILE : AkaSource.NONE);
  }

  /**
   * 鳴いた牌と手から出した牌から、チー・ポン・大明槓の面子を作る。赤の由来は、鳴いた牌が赤なら鳴いた牌、手から出した牌に赤が
   * あれば手の牌とする。牌がその種別の面子にならないとき、赤の 5 が 2 枚あるとき、上家以外からチーしたとき、暗槓・加槓を
   * 渡したときは、IllegalArgumentException を投げる。
   *
   * @param type 面子の種別
   * @param source 鳴き元
   * @param called 鳴いた牌の表記
   * @param consumed 手から出した牌の表記
   */
  public static Meld call(Type type, RelativeSource source, Face called, List<Face> consumed) {
    int[] kinds = new int[consumed.size() + 1];
    kinds[0] = called.kind();
    AkaSource aka = called.red() ? AkaSource.CALLED_TILE : AkaSource.NONE;
    for (int index = 0; index < consumed.size(); index++) {
      Face face = consumed.get(index);
      kinds[index + 1] = face.kind();
      if (face.red()) {
        if (aka != AkaSource.NONE) {
          throw new IllegalArgumentException("A meld cannot hold two red fives: " + called + " " + consumed);
        }
        aka = AkaSource.CONSUMED_HAND_TILE;
      }
    }
    Arrays.sort(kinds);
    Meld meld =
        switch (type) {
          case CHI -> {
            if (source != RelativeSource.KAMICHA) {
              throw new IllegalArgumentException("A chi is called from the left, not from " + source);
            }
            yield chi(kinds[0], called.kind(), aka);
          }
          case PON -> pon(called.kind(), source, aka);
          case DAIMINKAN -> daiminkan(called.kind(), source, aka);
          case ANKAN, KAKAN -> throw new IllegalArgumentException("Not a call: " + type);
        };
    boolean fits = kinds.length == meld.size();
    for (int index = 0; fits && index < kinds.length; index++) {
      fits = meld.tile(index) == kinds[index];
    }
    if (!fits) {
      throw new IllegalArgumentException("Tiles " + called + " " + consumed + " do not make " + meld);
    }
    return meld;
  }

  /**
   * ポンに 4 枚目を足した加槓の面子を返す。鳴き元と赤の由来は元のポンから引き継ぐ。ポンでない面子を渡したときと、赤の 5 が
   * 2 枚になるときは IllegalArgumentException を投げる。
   *
   * @param pon 元のポン
   * @param addedRed 足した 4 枚目が赤なら true
   */
  public static Meld kakan(Meld pon, boolean addedRed) {
    if (pon.type != Type.PON) {
      throw new IllegalArgumentException("KAKAN needs a PON, got " + pon);
    }
    if (addedRed && pon.red()) {
      throw new IllegalArgumentException("A meld cannot hold two red fives: " + pon);
    }
    return lookup(Type.KAKAN, pon.base, pon.called, pon.source, addedRed ? AkaSource.ADDED_KAN_TILE : pon.aka);
  }

  public int id() {
    return id;
  }

  public Type type() {
    return type;
  }

  /** チーは順子の最小の種類、ほかは面子の種類。 */
  public int base() {
    return base;
  }

  /** 鳴いた牌の種類。暗槓は −1。 */
  public int called() {
    return called;
  }

  public RelativeSource source() {
    return source;
  }

  public AkaSource aka() {
    return aka;
  }

  public int size() {
    return type.tiles;
  }

  /**
   * 面子の何枚目かの牌の種類を返す。
   *
   * @param index 何枚目か（0 始まり、順子は小さい順）
   */
  public int tile(int index) {
    if (index < 0 || index >= type.tiles) {
      throw new IndexOutOfBoundsException("Meld tile " + index + " of " + this);
    }
    return type == Type.CHI ? base + index : base;
  }

  /** 面子に含まれる牌の種類のビットマスク。 */
  public long kinds() {
    return type == Type.CHI ? Tiles.range(base, base + RUN - 1) : Tiles.bit(base);
  }

  /** 赤の 5 を含むなら true。 */
  public boolean red() {
    return aka != AkaSource.NONE;
  }

  /** 番号をそのまま返す。オブジェクトごとの既定のハッシュは実行のたびに変わり、HashMap などの並び順が揺れるので使わない。 */
  @Override
  public int hashCode() {
    return id;
  }

  @Override
  public boolean equals(Object other) {
    return this == other;
  }

  @Override
  public String toString() {
    StringBuilder text = new StringBuilder(type.name()).append('(');
    for (int index = 0; index < type.tiles; index++) {
      text.append(Face.of(tile(index), false).mjai());
    }
    if (called >= 0) {
      text.append(' ').append(source).append(' ').append(Face.of(called, false).mjai());
    }
    if (red()) {
      text.append(' ').append(aka);
    }
    return text.append(')').toString();
  }

  /**
   * 手の赤の 5 を使えるチーなら true。順子が 5 を含み、鳴いた位置の牌が 5 でないときに限る。
   *
   * @param start 順子が始まる数字（色の中で 0 始まり）
   * @param position 鳴いた位置
   */
  static boolean redFromHand(int start, int position) {
    return start <= Tiles.FIVE && Tiles.FIVE < start + RUN && start + position != Tiles.FIVE;
  }

  private static Meld lookup(Type type, int base, int called, RelativeSource source, AkaSource aka) {
    if (base < 0 || base >= Tiles.KINDS || type == Type.CHI && (called < base || called >= base + RUN)) {
      throw new IllegalArgumentException("Not a meld: " + type + " base " + base + " called " + called);
    }
    Meld meld = TABLE[idOf(type, base, called, source, aka)];
    if (meld == null) {
      throw new IllegalArgumentException(
          "Not a meld: " + type + " base " + base + " called " + called + " " + source + " " + aka);
    }
    return meld;
  }

  private static int idOf(Type type, int base, int called, RelativeSource source, AkaSource aka) {
    int position = type == Type.CHI ? called - base : 0;
    return (((type.ordinal() * Tiles.KINDS + base) * SOURCES.length + source.ordinal()) * RUN + position)
            * AKA_SOURCES.length
        + aka.ordinal();
  }

  /**
   * 天鳳で作れる面子だけを表に置く。チーは上家から、ポン・大明槓・加槓は他家から鳴く。暗槓は鳴き元なし。赤を含めるのは、5 を
   * 含む面子だけ。どの牌のセットでも 5 には赤があり、槓子は 4 枚を全て使うので、5 の槓子は必ず赤を含む。赤の無い 5 の槓子は
   * 表に置かない。
   */
  private static Meld[] table() {
    Meld[] table = new Meld[IDS];
    for (Tiles.Group suit : Tiles.Group.values()) {
      if (!suit.suited()) {
        continue;
      }
      for (int start = 0; start < RUN_STARTS; start++) {
        int base = suit.first() + start;
        for (int position = 0; position < RUN; position++) {
          int called = base + position;
          put(table, new Meld(Type.CHI, base, called, RelativeSource.KAMICHA, AkaSource.NONE));
          if (start + position == Tiles.FIVE) {
            put(table, new Meld(Type.CHI, base, called, RelativeSource.KAMICHA, AkaSource.CALLED_TILE));
          } else if (redFromHand(start, position)) {
            put(table, new Meld(Type.CHI, base, called, RelativeSource.KAMICHA, AkaSource.CONSUMED_HAND_TILE));
          }
        }
      }
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      boolean five = Tiles.contains(Tiles.FIVES, kind);
      AkaSource concealed = five ? AkaSource.CONSUMED_HAND_TILE : AkaSource.NONE;
      put(table, new Meld(Type.ANKAN, kind, -1, RelativeSource.NONE, concealed));
      for (RelativeSource source : OPEN_SOURCES) {
        for (Type type : OPEN_TRIPLETS) {
          if (!(five && type.kan())) {
            put(table, new Meld(type, kind, kind, source, AkaSource.NONE));
          }
          if (five) {
            put(table, new Meld(type, kind, kind, source, AkaSource.CALLED_TILE));
            put(table, new Meld(type, kind, kind, source, AkaSource.CONSUMED_HAND_TILE));
          }
        }
        if (five) {
          put(table, new Meld(Type.KAKAN, kind, kind, source, AkaSource.ADDED_KAN_TILE));
        }
      }
    }
    return table;
  }

  private static void put(Meld[] table, Meld meld) {
    if (table[meld.id] != null) {
      throw new IllegalStateException("Meld id " + meld.id + " is taken twice");
    }
    table[meld.id] = meld;
  }
}
