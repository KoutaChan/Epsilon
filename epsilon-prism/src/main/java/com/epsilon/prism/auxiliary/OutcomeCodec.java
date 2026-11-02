package com.epsilon.prism.auxiliary;

import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.core.GameState;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.RoundResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import com.epsilon.engine.WinClaim;
import com.epsilon.engine.WinPointDeltas;
import com.epsilon.prism.Limits;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.codec.BitField;
import com.epsilon.prism.codec.BitLayout;
import com.epsilon.prism.codec.SchemaDigest;
import com.epsilon.prism.observe.ObservationSchema;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 測った手の経路の局末を、判断の席から見た 1 枠 {@link #WORDS} 語に詰めて解く（補助の頭の候補の教師の記録）。詰めと解きはこのクラスだけが
 * 行い、語の並びは {@link Word}、語の中の配置は欄の enum から {@link BitLayout} が作る。
 *
 * <ul>
 *   <li>{@link Word#FACTS}: 局の終わり方・和了した他家・聴牌と役のある聴牌への到達・打点・失点・相手 3 席の局末の聴牌・次局（局・本場・
 *       供託。半荘が終われば局はなし）。
 *   <li>{@link Word#YAKU}: 関係する和了の役のビット（エンジンの役のビットそのまま、束ねない。ビット k は {@link ScoringYaku} の ordinal
 *       k）。自分の和了なら自分の和了、他家の和了（放銃を含む）なら {@link CandidateTarget#WINNER} と同じ最初の和了、流局は 0。
 *   <li>{@link Word#SCORES}: 局末の 4 席の点（判断の席から見た相対席の順）。局収支は、これから判断の時点の点（記録 O）を引いて作る。
 * </ul>
 *
 * <p>局の終わり方（{@link Outcome}）: ツモ和了は自分か他家か、ロン和了は自分が和了者の 1 人（複数ロンを含む）・放銃者（槍槓を含む）・
 * どちらでもないか、荒牌流局は自分の局末の聴牌（流し満貫も流局の扱いで、点は {@link Word#SCORES} が持つ）、途中流局。
 *
 * <p>打点は自分の和了で受け取る点、失点は放銃で払う点の和（複数ロン）で、どちらも本場・供託を含まない。和了の点は手の基本点から
 * {@link WinPointDeltas} で作る（WinPayment.total は本場を含むので使わない）。包のある和了の放銃は、エンジンの精算で自分が実際に払う額に
 * 合わせる。精算の点の移動で自分が本場込みの全額を払っていれば（包がないか、責任者が自分）本場を除いた全額、責任者が他家で半額だけ払って
 * いればその半額（本場は責任者が払う）。点は {@link ObservationSchema#POINT_UNIT} 点単位で書く。
 *
 * <p>記録のない枠（測っていない手と、候補の教師を使わない手）は全ての語が 0 で、{@link #outcome} は null を返す。書く口は点の移動の作業域を
 * 使うので、書くスレッド（局末の値を渡すドライバ）ごとに 1 つ持つ。解く口は static で、割り当てをしない。
 */
public final class OutcomeCodec {

  /** 1 枠の語の数。 */
  public static final int WORDS = Word.values().length;

  private static final Outcome[] OUTCOMES = Outcome.values();

  /** 判断の席から見た相手の数（下家・対面・上家）。 */
  private static final int OPPONENTS = GameState.NUM_PLAYERS - 1;

  /** 役のビットとして書ける範囲（{@link ScoringYaku} の数）。 */
  private static final long YAKU_BITS = (1L << ScoringYaku.values().length) - 1;

  /** {@link Word#FACTS} の配置。 */
  static final BitLayout<FactBits> FACTS = BitLayout.of(FactBits.class, Long.SIZE);

  /** {@link Word#SCORES} の配置。 */
  static final BitLayout<ScoreBits> SCORES = BitLayout.of(ScoreBits.class, Long.SIZE);

  static {
    if (FACTS.units() != 1 || SCORES.units() != 1) {
      throw new IllegalStateException("each outcome word must fit in one long");
    }
  }

  private final int[] deltas = new int[GameState.NUM_PLAYERS];

  /**
   * 局末の事実を、判断の席から見た 1 枠に書く。
   *
   * @param end 経路の局末の事実
   * @param seat 判断の席（絶対席）
   * @param events 判断の時点の出来事の数（記録 O の頭）
   * @param to 書く先
   * @param at 枠の先頭
   * @throws IllegalArgumentException 出来事の数が負か、値が欄の範囲の外のとき
   * @throws IllegalStateException 精算がこの記録の決まり（点の単位・包の支払い・役のビット）と合わないとき
   */
  public void write(HandEndFacts end, int seat, int events, long[] to, int at) {
    Objects.checkIndex(seat, GameState.NUM_PLAYERS);
    Objects.checkFromIndexSize(at, WORDS, to.length);
    Arrays.fill(to, at, at + WORDS, 0L);
    int facts = at + Word.FACTS.ordinal();
    RoundSettlement settlement = end.settlement();
    switch (settlement.result()) {
      case RoundResult.TsumoAgari tsumo -> writeTsumo(tsumo.claim(), seat, end.dealer(), to, at);
      case RoundResult.RonAgari ron -> writeRon(ron, seat, end.dealer(), to, at);
      case RoundResult.ExhaustiveDraw ignored -> {
        Outcome draw = end.tenpai(seat) ? Outcome.DRAW_TENPAI : Outcome.DRAW_NOTEN;
        put(to, facts, FactBits.OUTCOME, draw.ordinal());
      }
      case RoundResult.AbortiveDraw ignored ->
          put(to, facts, FactBits.OUTCOME, Outcome.ABORTIVE.ordinal());
    }
    put(to, facts, FactBits.TENPAI, end.reachedTenpai(seat, events) ? 1 : 0);
    put(to, facts, FactBits.YAKU_TENPAI, end.reachedYakuTenpai(seat, events) ? 1 : 0);
    for (int opponent = 1; opponent <= OPPONENTS; opponent++) {
      int tenpai = end.tenpai(absolute(seat, opponent)) ? 1 : 0;
      FACTS.put(to, facts, 0, FactBits.OPPONENT_TENPAI, opponent - 1, tenpai);
    }
    switch (settlement.transition()) {
      case RoundTransition.NextRound next -> {
        put(to, facts, FactBits.NEXT_KYOKU, next.kyokuIndex());
        put(to, facts, FactBits.NEXT_HONBA, next.honba());
        put(to, facts, FactBits.NEXT_KYOTAKU, next.kyotakuCount());
      }
      case RoundTransition.HanchanFinished ignored -> put(to, facts, FactBits.NEXT_KYOKU, -1);
    }
    int scores = at + Word.SCORES.ordinal();
    for (int relative = 0; relative < GameState.NUM_PLAYERS; relative++) {
      int points = settlement.finalScore(absolute(seat, relative));
      SCORES.put(to, scores, 0, ScoreBits.SCORE, relative, hundreds(points));
    }
  }

  private void writeTsumo(WinClaim.Tsumo claim, int seat, int dealer, long[] to, int at) {
    int facts = at + Word.FACTS.ordinal();
    if (claim.winner() == seat) {
      put(to, facts, FactBits.OUTCOME, Outcome.SELF_TSUMO.ordinal());
      put(to, facts, FactBits.WIN_POINTS, hundreds(winPoints(claim, dealer)));
    } else {
      put(to, facts, FactBits.OUTCOME, Outcome.OTHER_TSUMO.ordinal());
      put(to, facts, FactBits.WINNER, relative(seat, claim.winner()) - 1);
    }
    writeYaku(claim, to, at);
  }

  private void writeRon(RoundResult.RonAgari ron, int seat, int dealer, long[] to, int at) {
    int facts = at + Word.FACTS.ordinal();
    List<WinClaim.Ron> claims = ron.claims();
    for (int index = 0, size = claims.size(); index < size; index++) {
      WinClaim.Ron claim = claims.get(index);
      if (claim.winner() == seat) {
        put(to, facts, FactBits.OUTCOME, Outcome.SELF_RON.ordinal());
        put(to, facts, FactBits.WIN_POINTS, hundreds(winPoints(claim, dealer)));
        writeYaku(claim, to, at);
        return;
      }
    }
    WinClaim.Ron first = claims.getFirst();
    put(to, facts, FactBits.WINNER, relative(seat, first.winner()) - 1);
    writeYaku(first, to, at);
    if (ron.loser() != seat) {
      put(to, facts, FactBits.OUTCOME, Outcome.OTHER_RON.ordinal());
      return;
    }
    int paid = 0;
    for (int index = 0, size = claims.size(); index < size; index++) {
      paid += dealInPoints(claims.get(index), seat, dealer);
    }
    put(to, facts, FactBits.OUTCOME, Outcome.DEAL_IN.ordinal());
    put(to, facts, FactBits.DEAL_IN_POINTS, hundreds(paid));
  }

  /** 和了者が受け取る点（本場・供託を除く）。包は受け取りを変えない。 */
  private int winPoints(WinClaim claim, int dealer) {
    WinPointDeltas.settleInto(
        claim.score().basePoints(), claim.winner(), claim.from(), dealer, 0, 0, deltas);
    return deltas[claim.winner()];
  }

  /** ロン 1 件で放銃者 {@code seat} が実際に払う点（本場を除く）。決め方はクラスの説明のとおり。 */
  private int dealInPoints(WinClaim.Ron claim, int seat, int dealer) {
    int base = claim.score().basePoints();
    WinPointDeltas.settleInto(base, claim.winner(), seat, dealer, 0, 0, deltas);
    int full = -deltas[seat];
    WinPointDeltas.settleInto(base, claim.winner(), seat, dealer, claim.honba(), 0, deltas);
    int fullWithHonba = -deltas[seat];
    int paid = -claim.pointDelta().get(seat);
    if (paid == fullWithHonba) {
      return full;
    }
    if (paid == full / 2) {
      return paid;
    }
    throw new IllegalStateException(
        "deal-in payment " + paid + " is neither the full " + fullWithHonba + " nor the pao share");
  }

  private static void writeYaku(WinClaim claim, long[] to, int at) {
    long yaku = claim.score().yakuBits();
    if ((yaku & ~YAKU_BITS) != 0L) {
      throw new IllegalStateException(
          "yaku bits outside the scoring yaku: " + Long.toHexString(yaku));
    }
    to[at + Word.YAKU.ordinal()] = yaku;
  }

  /**
   * 局の終わり方を返す。
   *
   * @param words 枠を並べた配列
   * @param at 枠の先頭
   * @return 局の終わり方（記録のない枠は null）
   */
  public static Outcome outcome(long[] words, int at) {
    int ordinal = fact(words, at, FactBits.OUTCOME);
    return ordinal < 0 ? null : OUTCOMES[ordinal];
  }

  /**
   * 和了した他家の相対席を返す（{@link CandidateTarget#WINNER}）。
   *
   * @param words 枠を並べた配列
   * @param at 枠の先頭
   * @return 相対席（下家 1・対面 2・上家 3。他家の和了でなければ −1）
   */
  public static int winner(long[] words, int at) {
    int winner = fact(words, at, FactBits.WINNER);
    return winner < 0 ? -1 : winner + 1;
  }

  /** 判断の後、局の終わりまでに聴牌に届いた（和了も true）。 */
  public static boolean tenpai(long[] words, int at) {
    return fact(words, at, FactBits.TENPAI) != 0;
  }

  /** 判断の後、局の終わりまでにロンの役がある聴牌に届いた（和了も true）。 */
  public static boolean yakuTenpai(long[] words, int at) {
    return fact(words, at, FactBits.YAKU_TENPAI) != 0;
  }

  /** 自分の和了で受け取った点（本場・供託を除く。和了しなければ 0）。 */
  public static int winPoints(long[] words, int at) {
    return fact(words, at, FactBits.WIN_POINTS) * ObservationSchema.POINT_UNIT;
  }

  /** 放銃で払った点の和（本場を除く。放銃しなければ 0）。 */
  public static int dealInPoints(long[] words, int at) {
    return fact(words, at, FactBits.DEAL_IN_POINTS) * ObservationSchema.POINT_UNIT;
  }

  /**
   * 相手の局末の聴牌（和了を含む）を返す。
   *
   * @param words 枠を並べた配列
   * @param at 枠の先頭
   * @param opponent 相手の相対席（下家 1・対面 2・上家 3）
   * @return 聴牌なら true
   */
  public static boolean opponentTenpai(long[] words, int at, int opponent) {
    int index = Objects.checkIndex(opponent - 1, OPPONENTS);
    return FACTS.get(words, at + Word.FACTS.ordinal(), 0, FactBits.OPPONENT_TENPAI, index) != 0;
  }

  /** 次局の局（東 1 局 0 … 西 4 局 11。半荘が終われば −1）。 */
  public static int nextKyoku(long[] words, int at) {
    return fact(words, at, FactBits.NEXT_KYOKU);
  }

  /** この局で半荘が終わった（次局がない）。 */
  public static boolean hanchanEnded(long[] words, int at) {
    return nextKyoku(words, at) < 0;
  }

  /** 次局の本場（半荘が終われば 0）。 */
  public static int nextHonba(long[] words, int at) {
    return fact(words, at, FactBits.NEXT_HONBA);
  }

  /** 次局に持ち越す供託の本数（半荘が終われば 0）。 */
  public static int nextKyotaku(long[] words, int at) {
    return fact(words, at, FactBits.NEXT_KYOTAKU);
  }

  /** 関係する和了の役のビット（ビット k は {@link ScoringYaku} の ordinal k。流局は 0）。 */
  public static long yakuBits(long[] words, int at) {
    return words[at + Word.YAKU.ordinal()];
  }

  /**
   * 局末の点を返す。
   *
   * @param words 枠を並べた配列
   * @param at 枠の先頭
   * @param relative 判断の席から見た相対席（自分 0・下家 1・対面 2・上家 3）
   * @return 点
   */
  public static int finalScore(long[] words, int at, int relative) {
    return SCORES.get(words, at + Word.SCORES.ordinal(), 0, ScoreBits.SCORE, relative)
        * ObservationSchema.POINT_UNIT;
  }

  /**
   * 局末の記録の指紋を返す。行の目録に書き、読むときに照らす。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  private static int fact(long[] words, int at, FactBits field) {
    return FACTS.get(words, at + Word.FACTS.ordinal(), 0, field);
  }

  private static void put(long[] words, int facts, FactBits field, int value) {
    FACTS.put(words, facts, 0, field, value);
  }

  /** 判断の席から見た相対席の絶対席。 */
  private static int absolute(int seat, int relative) {
    return (seat + relative) % GameState.NUM_PLAYERS;
  }

  /** 判断の席から見た絶対席 {@code player} の相対席。 */
  private static int relative(int seat, int player) {
    return Math.floorMod(player - seat, GameState.NUM_PLAYERS);
  }

  /** 点を {@link ObservationSchema#POINT_UNIT} 点単位にする。 */
  private static int hundreds(int points) {
    if (points % ObservationSchema.POINT_UNIT != 0) {
      throw new IllegalStateException(
          "points are not a multiple of " + ObservationSchema.POINT_UNIT + ": " + points);
    }
    return points / ObservationSchema.POINT_UNIT;
  }

  /** 1 枠の語（宣言の順に並べる）。 */
  enum Word {
    /** 局末の事実（{@link FactBits}）。 */
    FACTS,
    /** 関係する和了の役のビット。 */
    YAKU,
    /** 局末の点（{@link ScoreBits}）。 */
    SCORES
  }

  /** {@link Word#FACTS} の欄。 */
  enum FactBits implements BitField {
    /** 局の終わり方（{@link Outcome} の ordinal。記録のない枠は −1）。 */
    OUTCOME(-1, Outcome.values().length - 1, 1),
    /** 和了した他家（相対席 − 1。他家の和了でなければ −1）。 */
    WINNER(-1, GameState.NUM_PLAYERS - 2, 1),
    /** 判断の後、局の終わりまでに聴牌に届いた（和了も 1）。 */
    TENPAI(0, 1, 1),
    /** 判断の後、局の終わりまでにロンの役がある聴牌に届いた（和了も 1）。 */
    YAKU_TENPAI(0, 1, 1),
    /** 自分の和了で受け取る点（本場・供託を除く）。 */
    WIN_POINTS(0, Limits.MAX_WIN_POINTS / ObservationSchema.POINT_UNIT, 1),
    /** 放銃で払う点の和（本場を除く）。 */
    DEAL_IN_POINTS(0, Limits.MAX_DEAL_IN_POINTS / ObservationSchema.POINT_UNIT, 1),
    /** 相手の局末の聴牌（和了を含む。繰り返しの番号は相対席 − 1）。 */
    OPPONENT_TENPAI(0, 1, GameState.NUM_PLAYERS - 1),
    /** 次局の局（半荘が終われば −1）。 */
    NEXT_KYOKU(-1, HanchanProgression.WEST_END_KYOKU, 1),
    /** 次局の本場（半荘が終われば 0。次局の O に入る値なので O と同じ上限）。 */
    NEXT_HONBA(0, ObservationSchema.MAX_HONBA, 1),
    /** 次局に持ち越す供託の本数（半荘が終われば 0）。 */
    NEXT_KYOTAKU(0, Limits.MAX_KYOTAKU, 1);

    private final int min;
    private final int max;
    private final int count;

    FactBits(int min, int max, int count) {
      this.min = min;
      this.max = max;
      this.count = count;
    }

    @Override
    public int min() {
      return min;
    }

    @Override
    public int max() {
      return max;
    }

    @Override
    public int count() {
      return count;
    }
  }

  /** {@link Word#SCORES} の欄。 */
  enum ScoreBits implements BitField {
    /** 局末の点（{@link ObservationSchema#POINT_UNIT} 点単位、符号つき 16 bit。繰り返しの番号は相対席）。 */
    SCORE;

    @Override
    public int min() {
      return Short.MIN_VALUE;
    }

    @Override
    public int max() {
      return Short.MAX_VALUE;
    }

    @Override
    public int count() {
      return GameState.NUM_PLAYERS;
    }
  }

  /** 遅延で作る（初期化の順の罠を避ける）。 */
  private static final class Fingerprint {

    private static final String VALUE =
        SchemaDigest.named("epsilon-prism-v2-outcome")
            .values(Word.class)
            .layout(FACTS)
            .layout(SCORES)
            .values(Outcome.class)
            .values(ScoringYaku.class)
            .with("pointUnit", ObservationSchema.POINT_UNIT)
            .hex();
  }
}
