package com.epsilon.classic.rulebot;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.classic.input.RoundRules;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 判断の時点で見えている事実。見えている牌の枚数・席ごとの事実・他家の脅威・巡目・ツモの残り・終盤の順位を、判断の記録だけから
 * 求め、打牌の危険度を出す。席は自分から見た相対席。判断のたびに {@link #load} で作り直し、メモリは確保しない。1 つのインスタンスを
 * 同時に使えるのは 1 スレッドだけ。
 *
 * <p>脅威は 2 種類ある。1 つは立直した他家で、宣言牌がまだ通っていない立直の宣言も含む。重みは 100、親なら 150。もう 1 つは、立直
 * していない他家のうち、副露が 3 つ以上か、12 巡目以降で 2 つ以上の他家。重みは 35 ＋ 3 つ目からの副露ごとに 20 ＋ 見えている面子の
 * 値ごとに 10 で、親なら 1.5 倍にし、100 を上限にする。面子の値は、三元牌・その席の自風・場風の刻子と槓子ごとに 1、赤ごとに 1、面子の
 * 牌ごとにドラの数。巡目は {@link RoundRules#afterDraw} と同じ数え方で、チー・ポンの後の打牌は数えない。
 *
 * <p>危険度は小さいほど安全で、脅威ごとの値に、その脅威の重み ÷ 100 を掛けて足したもの。脅威ごとの値は、次の表の値にドラの分を足す。
 * ドラの分は、ドラ 1 つにつき 15、数牌ではさらに隣の数のドラ 1 つにつき 5。現物は、立直した他家なら河と立直の後に通った牌、ほかの
 * 他家なら河の牌。筋は、立直した他家の、立直より前の河だけで見る。見えている枚数には自分の純手牌も含める。
 *
 * <table>
 *   <caption>脅威ごとの値</caption>
 *   <tr><td>0</td><td>現物</td></tr>
 *   <tr><td>5・20・40</td><td>字牌で、見えている枚数が 4 枚・2 枚以上・それより少ない</td></tr>
 *   <tr><td>50・70・80・90</td><td>数牌の 1 と 9・2 と 8・3 と 7・4〜6</td></tr>
 *   <tr><td>30 まで</td><td>筋（4〜6 は両筋）か、その牌を使う順子が全て壁で塞がる（ノーチャンス）</td></tr>
 *   <tr><td>50 まで</td><td>片筋</td></tr>
 *   <tr><td>40・50 まで</td><td>その牌を使うどの順子も、ほかの 2 枚のどちらかが 3 枚見えている（ワンチャンス）。山の残りが多ければ 40、少なければ 50</td></tr>
 * </table>
 */
final class Situation {

  private static final int DANGER_GENBUTSU = 0;
  private static final int DANGER_FOURTH_HONOR = 5;
  private static final int DANGER_SEEN_HONOR = 20;
  private static final int DANGER_SUJI_OR_NO_CHANCE = 30;
  private static final int DANGER_LIVE_HONOR = 40;
  private static final int DANGER_EARLY_ONE_CHANCE = 40;
  private static final int DANGER_TERMINAL = 50;
  private static final int DANGER_LATE_ONE_CHANCE = 50;
  private static final int DANGER_TWO_EIGHT = 70;
  private static final int DANGER_THREE_SEVEN = 80;
  private static final int DANGER_MIDDLE = 90;
  private static final int DANGER_PER_DORA = 15;
  private static final int DANGER_PER_NEIGHBOR_DORA = 5;

  /** 字牌の値を 20 に下げる、見えている枚数の下限（4 枚見えていれば 5）。 */
  private static final int SEEN_HONOR = 2;

  private static final int RIICHI_WEIGHT = 100;
  private static final int DEALER_RIICHI_WEIGHT = 150;
  private static final int OPEN_WEIGHT = 35;
  private static final int OPEN_WEIGHT_PER_MELD = 20;
  private static final int OPEN_WEIGHT_PER_VALUE = 10;
  private static final int MAX_WEIGHT = 100;
  private static final int PERCENT = 100;

  /** 立直していない他家を脅威と見る副露の数（巡目に関係なく脅威と見る数と、12 巡目以降に脅威と見る数）。 */
  private static final int OPEN_THREAT_MELDS = 3;

  private static final int LATE_OPEN_THREAT_MELDS = 2;

  /** 副露が 2 つの他家を脅威と見始める巡目。 */
  private static final int LATE_OPEN_THREAT_TURN = 12;

  /** 筋の間隔（数牌の数の差）。 */
  private static final int SUJI = 3;

  /** 順子の枚数。 */
  private static final int RUN = 3;

  /** 1 色の中で、最後の順子が始まる番号（7 の牌の番号）。 */
  private static final int LAST_RUN_START = Tiles.SUIT_SIZE - RUN;

  private final Variant variant;
  private final int seats;
  private final TileSet set;
  private final TileCounts tiles;
  private final SeatFacts facts;
  private final Observation.EventRecord record = new Observation.EventRecord();

  /** ワンチャンスを 40 と見る、山の残りの下限（山の 4/7 で、四人打ちでは 40 枚）。 */
  private final int reliableOneChanceWall;

  /** 相対席ごとに、立直の宣言牌より前の河にある牌の種類。 */
  private final long[] beforeRiichi;

  private final int[] threatWeight;
  private final long[] threatSafe;
  private final long[] threatSuji;
  private int threats;
  private int riichiThreats;
  private int turn;
  private int selfDrawsLeft;
  private int wall;
  private int seatWind;
  private int roundWind;
  private boolean dealer;
  private boolean lateFirst;
  private boolean lateLast;

  /** 見逃しの振聴中で、自分が打牌した後もそれが残るか。 */
  private boolean missedRonStays;

  private RiichiState riichi;
  private RiichiState declaration;

  Situation(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    set = variant.tiles();
    tiles = new TileCounts(variant);
    facts = new SeatFacts(variant);
    reliableOneChanceWall = variant.liveWall() * 4 / 7;
    beforeRiichi = new long[seats];
    threatWeight = new int[seats - 1];
    threatSafe = new long[seats - 1];
    threatSuji = new long[seats - 1];
  }

  /** 判断の時点の事実を作る。 */
  void load(Observation o) {
    tiles.load(o);
    facts.load(o);
    int round = o.kyoku();
    seatWind = variant.seatWind(round, o.observer());
    roundWind = o.roundWindTile();
    dealer = o.dealer() == 0;
    wall = o.remainingWall();
    selfDrawsLeft = (wall + seats - 1) / seats;
    boolean late = round >= variant.lastRound() - 1;
    int rank = facts.rank(0);
    lateFirst = late && rank == 0;
    lateLast = late && rank == seats - 1;
    riichi = o.riichi(0);
    missedRonStays = o.furitenTemporary() && (riichi != RiichiState.NONE || RoundRules.afterCallFromShimocha(o));
    declaration = RoundRules.declaration(o.beforeFirstDiscard(0), o.callMade());
    scanDiscards(o);
    collectThreats(o);
  }

  TileCounts tiles() {
    return tiles;
  }

  SeatFacts facts() {
    return facts;
  }

  /** 脅威の数。 */
  int threats() {
    return threats;
  }

  /** 立直した他家の数。 */
  int riichiThreats() {
    return riichiThreats;
  }

  /** 巡目（ツモの後の打牌の数 ÷ 席の数、切り捨て）。 */
  int turn() {
    return turn;
  }

  /** 自分に残るツモの数の見込み（山の残り ÷ 席の数を切り上げた値）。 */
  int selfDrawsLeft() {
    return selfDrawsLeft;
  }

  int seatWind() {
    return seatWind;
  }

  int roundWind() {
    return roundWind;
  }

  boolean dealer() {
    return dealer;
  }

  /** 最後の 2 局で、持ち点の順位が 1 位か。 */
  boolean lateFirst() {
    return lateFirst;
  }

  /** 最後の 2 局で、持ち点の順位が最下位か。 */
  boolean lateLast() {
    return lateLast;
  }

  /** 自分の今の立直の状態。 */
  RiichiState riichi() {
    return riichi;
  }

  /** この打牌で立直を宣言したときの立直の状態。鳴きで途切れていない第一巡の、最初の打牌ならダブル立直。 */
  RiichiState declaration() {
    return declaration;
  }

  /** 自分にとっての役牌（三元牌・自風・場風）か。 */
  boolean yakuhai(int kind) {
    return Tiles.Group.of(kind) == Tiles.Group.DRAGON || kind == seatWind || kind == roundWind;
  }

  /**
   * 打牌の後に振聴になるか。見逃しの振聴が打牌の後も残るか、形の上の和了牌が自分の河（これから打つ牌を含む）にあれば振聴。
   * 見逃しの振聴は、立直していなければ自分の打牌で消える。残るのは、立直中と、下家から鳴いた後の最初の打牌だけ。
   *
   * @param waits 打牌の後の形の上の和了牌
   * @param discarded 打つ牌の種類
   */
  boolean furiten(long waits, int discarded) {
    return missedRonStays || (waits & (facts.riverKinds(0) | Tiles.bit(discarded))) != 0;
  }

  /**
   * 打つ牌の危険度を返す。脅威ごとの値に重みを付けて足したもので、脅威が無ければ 0。
   *
   * @param kind 打つ牌の種類
   */
  int danger(int kind) {
    int total = 0;
    for (int threat = 0; threat < threats; threat++) {
      total += against(kind, threat) * threatWeight[threat] / PERCENT;
    }
    return total;
  }

  /** ツモの後の打牌を数えて巡目を出す。立直を宣言した席については、宣言牌より前の河を集める。 */
  private void scanDiscards(Observation o) {
    Arrays.fill(beforeRiichi, 0);
    int drawn = 0;
    for (int event = 0; event < o.events(); event++) {
      o.readEvent(event, record);
      if (record.kind() != EventKind.DISCARD) {
        continue;
      }
      if (RoundRules.afterDraw(o, event)) {
        drawn++;
      }
      int author = record.author();
      if (event < facts.riichiDeclaration(author)) {
        beforeRiichi[author] |= Tiles.bit(record.discardTile());
      }
    }
    turn = drawn / seats;
  }

  /** 他家の脅威を集める。宣言牌がまだ通っていない立直の宣言も、立直として扱う。 */
  private void collectThreats(Observation o) {
    threats = 0;
    riichiThreats = 0;
    for (int seat = 1; seat < seats; seat++) {
      boolean seatDealer = o.dealer() == seat;
      if (o.riichi(seat) != RiichiState.NONE || facts.riichiDeclaration(seat) >= 0) {
        addThreat(seatDealer ? DEALER_RIICHI_WEIGHT : RIICHI_WEIGHT, facts.safe(seat), beforeRiichi[seat]);
        riichiThreats++;
        continue;
      }
      int melds = facts.meldCount(seat);
      if (melds < OPEN_THREAT_MELDS && (melds < LATE_OPEN_THREAT_MELDS || turn < LATE_OPEN_THREAT_TURN)) {
        continue;
      }
      int weight =
          OPEN_WEIGHT
              + Math.max(0, melds - LATE_OPEN_THREAT_MELDS) * OPEN_WEIGHT_PER_MELD
              + meldValue(o, seat) * OPEN_WEIGHT_PER_VALUE;
      if (seatDealer) {
        weight = weight * 3 / 2;
      }
      addThreat(Math.min(weight, MAX_WEIGHT), facts.riverKinds(seat), 0);
    }
  }

  private void addThreat(int weight, long safe, long suji) {
    threatWeight[threats] = weight;
    threatSafe[threats] = safe;
    threatSuji[threats] = suji;
    threats++;
  }

  /**
   * 他家の見えている面子の値を返す。数え方はクラスの説明のとおり。
   *
   * @param seat 他家の相対席
   */
  private int meldValue(Observation o, int seat) {
    int ownWind = variant.seatWind(o.kyoku(), o.absoluteSeat(seat));
    int value = 0;
    for (int index = 0; index < facts.meldCount(seat); index++) {
      Meld meld = facts.meld(seat, index);
      if (meld.type() != Meld.Type.CHI) {
        int kind = meld.base();
        value += Tiles.Group.of(kind) == Tiles.Group.DRAGON ? 1 : 0;
        value += kind == ownWind ? 1 : 0;
        value += kind == roundWind ? 1 : 0;
      }
      value += meld.red() ? 1 : 0;
      for (int tile = 0; tile < meld.size(); tile++) {
        value += tiles.dora(meld.tile(tile));
      }
    }
    return value;
  }

  private int against(int kind, int threat) {
    if (Tiles.contains(threatSafe[threat], kind)) {
      return DANGER_GENBUTSU;
    }
    int known = known(kind);
    if (!Tiles.Group.of(kind).suited()) {
      int danger =
          known >= Tiles.COPIES ? DANGER_FOURTH_HONOR : known >= SEEN_HONOR ? DANGER_SEEN_HONOR : DANGER_LIVE_HONOR;
      return danger + doraDanger(kind);
    }
    int number = Tiles.number(kind);
    int danger =
        switch (number) {
          case 0, 8 -> DANGER_TERMINAL;
          case 1, 7 -> DANGER_TWO_EIGHT;
          case 2, 6 -> DANGER_THREE_SEVEN;
          default -> DANGER_MIDDLE;
        };
    int anchors = sujiAnchors(kind, number, threatSuji[threat]);
    boolean middle = number >= SUJI && number < Tiles.SUIT_SIZE - SUJI;
    if (anchors >= (middle ? 2 : 1) || blocked(kind, number, Tiles.COPIES)) {
      danger = Math.min(danger, DANGER_SUJI_OR_NO_CHANCE);
    } else if (anchors > 0) {
      danger = Math.min(danger, DANGER_LATE_ONE_CHANCE);
    } else if (blocked(kind, number, Tiles.COPIES - 1)) {
      danger = Math.min(danger, wall >= reliableOneChanceWall ? DANGER_EARLY_ONE_CHANCE : DANGER_LATE_ONE_CHANCE);
    }
    return danger + doraDanger(kind);
  }

  /** 自分から見えている枚数（河・面子・表示牌と自分の純手牌）。 */
  private int known(int kind) {
    return tiles.visible(kind) + tiles.hand(kind);
  }

  private int doraDanger(int kind) {
    int danger = tiles.dora(kind) * DANGER_PER_DORA;
    if (!Tiles.Group.of(kind).suited()) {
      return danger;
    }
    int number = Tiles.number(kind);
    if (number > 0) {
      danger += tiles.dora(kind - 1) * DANGER_PER_NEIGHBOR_DORA;
    }
    if (number < Tiles.SUIT_SIZE - 1) {
      danger += tiles.dora(kind + 1) * DANGER_PER_NEIGHBOR_DORA;
    }
    return danger;
  }

  /** 筋になる牌（3 つ下と 3 つ上の数）のうち、筋を見る河にあるものの数。 */
  private static int sujiAnchors(int kind, int number, long anchors) {
    int count = 0;
    if (number + SUJI < Tiles.SUIT_SIZE && Tiles.contains(anchors, kind + SUJI)) {
      count++;
    }
    if (number >= SUJI && Tiles.contains(anchors, kind - SUJI)) {
      count++;
    }
    return count;
  }

  /**
   * その数牌を使う順子が全て塞がっているか。順子は、ほかの 2 枚のどちらかが、ルールで使わない牌か、見えている枚数が下限以上なら
   * 塞がっている。
   *
   * @param kind 数牌の種類
   * @param number 色の中の番号（0 始まり）
   * @param threshold 塞がっていると見る、見えている枚数の下限（ノーチャンスは 4、ワンチャンスは 3）
   */
  private boolean blocked(int kind, int number, int threshold) {
    for (int start = Math.max(0, number - (RUN - 1)); start <= Math.min(number, LAST_RUN_START); start++) {
      boolean closed = false;
      for (int offset = 0; offset < RUN && !closed; offset++) {
        int other = kind - number + start + offset;
        closed = other != kind && (set.copies(other) == 0 || known(other) >= threshold);
      }
      if (!closed) {
        return false;
      }
    }
    return true;
  }
}
