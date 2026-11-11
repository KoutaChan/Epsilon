package com.epsilon.mahjong.rules;

import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.EnumSet;

/**
 * ルール（四人打ち・三人打ち）。エンジン・牌譜・GRP・クライアントは、この値だけで四人打ちと三人打ちを分ける。
 * 席は起家を 0 とする絶対席。局の番号は 場風 × 席の数 ＋ 局（東 1 局が 0）で、親の席は 局の番号 mod 席の数。
 */
public enum Variant {
  TENHOU_4P(
      4,
      TileSet.FULL,
      25_000,
      30_000,
      EnumSet.of(RuleFlag.CHI),
      EnumSet.of(
          AbortiveDraw.NINE_TERMINALS,
          AbortiveDraw.FOUR_WINDS,
          AbortiveDraw.FOUR_RIICHI,
          AbortiveDraw.FOUR_KANS,
          AbortiveDraw.TRIPLE_RON)),
  TENHOU_3P(
      3,
      TileSet.SANMA,
      35_000,
      40_000,
      EnumSet.of(RuleFlag.NUKI, RuleFlag.NORTH_CALLS),
      EnumSet.of(AbortiveDraw.NINE_TERMINALS, AbortiveDraw.FOUR_KANS));

  /** 王牌の枚数。 */
  public static final int DEAD_WALL = 14;

  /** 配牌の枚数。 */
  public static final int HAND = 13;

  /** 供託 1 本の点（立直の宣言料）。 */
  public static final int DEPOSIT = 1000;

  /** 本場 1 本につき、ツモで払う各席が出す点。ロンでは放銃者が（席の数 − 1）人分を払う。 */
  public static final int HONBA = 100;

  /** 1 局の槓の上限。 */
  public static final int MAX_KANS = 4;

  /** 1 席の面子（副露と暗槓）の上限。 */
  public static final int MAX_MELDS = 4;

  /** 不聴罰符の総額は、この点 × (席の数 − 1)。 */
  private static final int NOTEN_PER_SEAT = 1000;

  private final int seats;
  private final TileSet tiles;
  private final int startPoints;
  private final int returnPoints;
  private final EnumSet<RuleFlag> flags;
  private final EnumSet<AbortiveDraw> aborts;
  private final long meldableKinds;
  private final int liveWall;
  private final int rinshanTiles;
  private final int maxDiscards;
  private final int maxDrawsPerSeat;
  private final int maxPublicEvents;

  Variant(
      int seats,
      TileSet tiles,
      int startPoints,
      int returnPoints,
      EnumSet<RuleFlag> flags,
      EnumSet<AbortiveDraw> aborts) {
    this.seats = seats;
    this.tiles = tiles;
    this.startPoints = startPoints;
    this.returnPoints = returnPoints;
    this.flags = flags;
    this.aborts = aborts;
    boolean northCalls = !flags.contains(RuleFlag.NUKI) || flags.contains(RuleFlag.NORTH_CALLS);
    meldableKinds = northCalls ? tiles.kinds() : tiles.kinds() & ~Tiles.bit(Tiles.PEI);
    int nukiTiles = flags.contains(RuleFlag.NUKI) ? Tiles.COPIES : 0;
    liveWall = tiles.size() - DEAD_WALL - HAND * seats;
    rinshanTiles = MAX_KANS + nukiTiles;
    maxDiscards = liveWall + MAX_MELDS * seats;
    maxDrawsPerSeat = Math.ceilDiv(liveWall, seats);
    maxPublicEvents = maxDiscards + MAX_MELDS * seats + MAX_KANS + nukiTiles;
  }

  public int seats() {
    return seats;
  }

  /** 全ての席のビットを立てたマスク（ビット s が席 s）。 */
  public int seatMask() {
    return (1 << seats) - 1;
  }

  public TileSet tiles() {
    return tiles;
  }

  public int startPoints() {
    return startPoints;
  }

  /** 返し点。終局の判定とオカに使う。 */
  public int returnPoints() {
    return returnPoints;
  }

  public boolean has(RuleFlag flag) {
    return flags.contains(flag);
  }

  /** このルールで起きる途中流局か。 */
  public boolean aborts(AbortiveDraw draw) {
    return aborts.contains(draw);
  }

  /** 面子（副露と暗槓）にできる牌の種類のマスク。北抜きのあるルールで北を鳴けないなら、北を除く。 */
  public long meldableKinds() {
    return meldableKinds;
  }

  /**
   * 山の枚数。牌の総数から王牌と配牌を引いた数（四人打ち 70 枚・三人打ち 55 枚）。槓か抜きを 1 回するたびに山が 1 枚減るので、
   * 嶺上牌を含めたツモの総数でもある。
   */
  public int liveWall() {
    return liveWall;
  }

  /** 嶺上牌として引ける枚数。槓の 4 枚に、北抜きのあるルールでは北の 4 枚を足す。 */
  public int rinshanTiles() {
    return rinshanTiles;
  }

  /**
   * 1 局の打牌の数の上限（四人打ち 86・三人打ち 67）。打牌が起きるのは、ツモの後（嶺上牌を含めて山の枚数以下）か、チー・ポンの後
   * （1 席 4 回以下）だけ。全ての席の打牌を合わせた数で、1 席の河の上限（山の枚数 ＋ 4）とは違う。
   */
  public int maxDiscards() {
    return maxDiscards;
  }

  /**
   * 山の牌を手番の順に 1 枚ずつ配ったときに、1 席が受け取る枚数の上限（四人打ち 18・三人打ち 19）。山の枚数を席の数で割って切り上げた
   * 数で、席ごとのツモ列の長さの上限でもある。鳴きで手番が飛ぶと、1 席が実際にツモる数はこれを超えることがある。
   */
  public int maxDrawsPerSeat() {
    return maxDrawsPerSeat;
  }

  /** ロンで本場 1 本につき放銃者が払う額: 100 × (席の数 − 1)。 */
  public int ronHonba() {
    return HONBA * (seats - 1);
  }

  /** 不聴罰符の総額: 1000 × (席の数 − 1)。 */
  public int notenTotal() {
    return NOTEN_PER_SEAT * (seats - 1);
  }

  /** 立直するのに要る、山の残り枚数の最小値（席の数と同じ）。 */
  public int riichiMinWall() {
    return seats;
  }

  /** 南場の最後の局の番号（2 × 席の数 − 1）。 */
  public int lastRound() {
    return 2 * seats - 1;
  }

  /** 西場の最後の局の番号（3 × 席の数 − 1）。この局で親が流れたら半荘は終わる。北場は無い。 */
  public int finalRound() {
    return 3 * seats - 1;
  }

  /**
   * 1 局で公開されるイベントの数の上限（四人打ち 106・三人打ち 87）。打牌の上限に、面子（1 席 4 つ）・加槓（4 つ）・抜き（北の 4 枚）を
   * 足した数で、イベントの位置は byte に収まる。
   */
  public int maxPublicEvents() {
    return maxPublicEvents;
  }

  public int dealer(int round) {
    return round % seats;
  }

  /** 場風の牌の種類（東・南・西）。 */
  public int roundWind(int round) {
    return Tiles.TON + round / seats;
  }

  /** 自風の牌の種類。親が東。 */
  public int seatWind(int round, int seat) {
    return Tiles.TON + Math.floorMod(seat - dealer(round), seats);
  }
}
