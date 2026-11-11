package com.epsilon.prism.auxiliary;

import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.agent.observe.codec.BitField;
import com.epsilon.agent.observe.codec.BitLayout;
import com.epsilon.agent.observe.codec.SchemaDigest;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.prism.Limits;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import java.util.Arrays;
import java.util.Objects;

/**
 * 測った手を打った後、その経路で局がどう終わったか（局末）を、判断した席から見て {@link #WORDS} ワードに詰め、またそこから読み出す。
 * 補助のヘッドが候補ごとに学ぶ教師の記録で、判断 1 件につき 1 件書く。詰めるのも読むのもこのクラスだけが行う。ワードの並びは
 * {@link Word} が決める。ワードの中の配置は、フィールドの enum からルールごとに {@link BitLayout} が作る。
 *
 * <ul>
 *   <li>{@link Word#FACTS}: 局の終わり方、和了した他家、聴牌と役のある聴牌に届いたか、打点、失点、相手の局末の聴牌、次の局（局・本場・
 *       供託。半荘が終われば次の局は無い）。
 *   <li>{@link Word#YAKU}: 和了の役のビット。エンジンの役のビットをまとめずにそのまま書き、ビット k が {@link ScoringYaku} の ordinal k
 *       の役。自分が和了すれば自分の役を、他家が和了すれば（放銃を含む）{@link CandidateTarget#WINNER} と同じ最初の和了者の役を
 *       書く。流局は 0。
 *   <li>{@link Word#SCORES}: 局末の全員の点。判断した席から見た相対席の順に並べる。局収支は、ここから判断の時点の点（記録 O にある）を
 *       引いて求める。
 * </ul>
 *
 * <p>局の終わり方（{@link Outcome}）は次のとおり。ツモ和了は、自分か他家か。ロン和了は、自分が和了者の 1 人か（複数ロンを含む）、放銃者か
 * （槍槓を含む）、どちらでもないか。荒牌流局は、自分が局末に聴牌かどうか。流し満貫も流局として扱い、その点は {@link Word#SCORES} が持つ。
 * 残りは途中流局。
 *
 * <p>打点は自分の和了で受け取る点で、本場 0・包なしで精算したときの額にする（三人打ちのツモ損を含む）。失点は放銃で払う点の和で、
 * 複数ロンなら全員の分を足し、本場は含めない。包のある和了に放銃したときは、その和了の分は半額にする（残りは包の席が払う）。点は
 * {@link ObservationSchema#POINT_UNIT} 点単位で書く。局末の点は精算の後の卓の点で、半荘が終わったときは残った供託をトップに渡した後の点。
 *
 * <p>記録のない判断（測っていない手と、候補の教師を使わない手）は全てのワードが 0 で、{@link #outcome} は null を返す。形式はルールごとに
 * 1 つだけある。どのスレッドからも使え、メモリを割り当てない。
 */
public final class OutcomeCodec {

  /** 判断 1 件分のワードの数。 */
  public static final int WORDS = Word.values().length;

  private static final Outcome[] OUTCOMES = Outcome.values();

  /** 役のビットとして書けるビットのマスク。下から {@link ScoringYaku} の数だけのビットが立つ。 */
  private static final long YAKU_BITS = (1L << ScoringYaku.values().length) - 1;

  private static final OutcomeCodec[] CODECS = new OutcomeCodec[Variant.values().length];

  static {
    for (Variant variant : Variant.values()) {
      CODECS[variant.ordinal()] = new OutcomeCodec(variant);
    }
  }

  private final Variant variant;
  private final int seats;

  /** {@link Word#FACTS} の配置。 */
  private final BitLayout<FactBits> facts;

  /** {@link Word#SCORES} の配置。 */
  private final BitLayout<ScoreBits> scores;

  private final String fingerprint;

  private OutcomeCodec(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    facts = BitLayout.of(FactBits.class, Long.SIZE, variant);
    scores = BitLayout.of(ScoreBits.class, Long.SIZE, variant);
    if (facts.units() != 1 || scores.units() != 1) {
      throw new IllegalStateException("Each outcome word must fit in one long for " + variant);
    }
    fingerprint =
        SchemaDigest.named("epsilon-prism-v2-outcome")
            .values(Vocabulary.of(Word.class, variant))
            .layout(facts)
            .layout(scores)
            .values(Vocabulary.of(Outcome.class, variant))
            .values(Vocabulary.of(ScoringYaku.class, variant))
            .with("pointUnit", ObservationSchema.POINT_UNIT)
            .hex();
  }

  /**
   * ルールごとの局末の記録の形式を返す。
   *
   * @param variant ルール
   * @return そのルールの形式
   */
  public static OutcomeCodec of(Variant variant) {
    return CODECS[variant.ordinal()];
  }

  public Variant variant() {
    return variant;
  }

  /**
   * 局末の事実を、判断した席から見た 1 件の記録として書く。
   *
   * @param end その経路の局末の事実
   * @param seat 判断した席（絶対席）
   * @param events 判断の時点までのイベントの数（記録 O のヘッダーにある値）
   * @param to 書き込む配列
   * @param at 記録の先頭の位置
   * @throws IllegalArgumentException ルールが違うか、イベントの数が負か、値がフィールドの範囲の外のとき
   * @throws IllegalStateException 点が {@link ObservationSchema#POINT_UNIT} 点の倍数でないか、{@link ScoringYaku} に無い役のビットが
   *     立っているとき
   */
  public void write(HandEndFacts end, int seat, int events, long[] to, int at) {
    if (end.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + end.variant());
    }
    Objects.checkIndex(seat, seats);
    Objects.checkFromIndexSize(at, WORDS, to.length);
    Arrays.fill(to, at, at + WORDS, 0L);
    int factsAt = at + Word.FACTS.ordinal();
    RoundOutcome outcome = end.outcome();
    int dealer = end.dealer();
    switch (outcome.kind()) {
      case TSUMO -> {
        int winner = outcome.winner(0);
        if (winner == seat) {
          put(to, factsAt, FactBits.OUTCOME, Outcome.SELF_TSUMO.ordinal());
          put(to, factsAt, FactBits.WIN_POINTS, hundreds(tsumoGain(outcome.base(0), winner, dealer)));
        } else {
          put(to, factsAt, FactBits.OUTCOME, Outcome.OTHER_TSUMO.ordinal());
          put(to, factsAt, FactBits.WINNER, relative(seat, winner) - 1);
        }
        writeYaku(end.yaku()[0], to, at);
      }
      case RON -> writeRon(end, seat, to, at);
      case EXHAUSTIVE -> {
        Outcome draw = end.tenpai(seat) ? Outcome.DRAW_TENPAI : Outcome.DRAW_NOTEN;
        put(to, factsAt, FactBits.OUTCOME, draw.ordinal());
      }
      case ABORTIVE -> put(to, factsAt, FactBits.OUTCOME, Outcome.ABORTIVE.ordinal());
    }
    put(to, factsAt, FactBits.TENPAI, end.reachedTenpai(seat, events) ? 1 : 0);
    put(to, factsAt, FactBits.YAKU_TENPAI, end.reachedYakuTenpai(seat, events) ? 1 : 0);
    for (int opponent = 1; opponent < seats; opponent++) {
      int tenpai = end.tenpai(absolute(seat, opponent)) ? 1 : 0;
      facts.put(to, factsAt, 0, FactBits.OPPONENT_TENPAI, opponent - 1, tenpai);
    }
    TableState next = end.next();
    if (next.gameOver()) {
      put(to, factsAt, FactBits.NEXT_KYOKU, -1);
    } else {
      put(to, factsAt, FactBits.NEXT_KYOKU, next.round());
      put(to, factsAt, FactBits.NEXT_HONBA, next.honba());
      put(to, factsAt, FactBits.NEXT_KYOTAKU, next.kyotaku());
    }
    int scoresAt = at + Word.SCORES.ordinal();
    for (int relative = 0; relative < seats; relative++) {
      scores.put(to, scoresAt, 0, ScoreBits.SCORE, relative, hundreds(next.score(absolute(seat, relative))));
    }
  }

  private void writeRon(HandEndFacts end, int seat, long[] to, int at) {
    int factsAt = at + Word.FACTS.ordinal();
    RoundOutcome outcome = end.outcome();
    int dealer = end.dealer();
    for (int index = 0, count = outcome.winnerCount(); index < count; index++) {
      int winner = outcome.winner(index);
      if (winner == seat) {
        put(to, factsAt, FactBits.OUTCOME, Outcome.SELF_RON.ordinal());
        put(to, factsAt, FactBits.WIN_POINTS, hundreds(ScoreMath.ron(outcome.base(index), winner == dealer)));
        writeYaku(end.yaku()[index], to, at);
        return;
      }
    }
    put(to, factsAt, FactBits.WINNER, relative(seat, outcome.winner(0)) - 1);
    writeYaku(end.yaku()[0], to, at);
    if (outcome.loser() != seat) {
      put(to, factsAt, FactBits.OUTCOME, Outcome.OTHER_RON.ordinal());
      return;
    }
    int paid = 0;
    for (int index = 0, count = outcome.winnerCount(); index < count; index++) {
      int full = ScoreMath.ron(outcome.base(index), outcome.winner(index) == dealer);
      paid += outcome.pao(index) == RoundOutcome.NO_SEAT ? full : full / 2;
    }
    put(to, factsAt, FactBits.OUTCOME, Outcome.DEAL_IN.ordinal());
    put(to, factsAt, FactBits.DEAL_IN_POINTS, hundreds(paid));
  }

  /** ツモの和了者が本場 0 で受け取る点。卓にいるほかの席が払う点の和。 */
  private int tsumoGain(int base, int winner, int dealer) {
    int gain = 0;
    for (int payer = 0; payer < seats; payer++) {
      if (payer != winner) {
        gain += ScoreMath.tsumo(base, winner == dealer, payer == dealer);
      }
    }
    return gain;
  }

  private static void writeYaku(long yaku, long[] to, int at) {
    if ((yaku & ~YAKU_BITS) != 0L) {
      throw new IllegalStateException("Yaku bits outside the scoring yaku: " + Long.toHexString(yaku));
    }
    to[at + Word.YAKU.ordinal()] = yaku;
  }

  /**
   * 局の終わり方を返す。
   *
   * @param words 記録を並べた配列
   * @param at 記録の先頭の位置
   * @return 局の終わり方。記録のない判断は null
   */
  public Outcome outcome(long[] words, int at) {
    int ordinal = fact(words, at, FactBits.OUTCOME);
    return ordinal < 0 ? null : OUTCOMES[ordinal];
  }

  /**
   * 和了した他家の相対席を返す（{@link CandidateTarget#WINNER}）。
   *
   * @param words 記録を並べた配列
   * @param at 記録の先頭の位置
   * @return 相対席（下家が 1）。他家が和了していなければ −1
   */
  public int winner(long[] words, int at) {
    int winner = fact(words, at, FactBits.WINNER);
    return winner < 0 ? -1 : winner + 1;
  }

  /** 判断の後、局の終わりまでに聴牌に届いたなら true を返す。和了も true。 */
  public boolean tenpai(long[] words, int at) {
    return fact(words, at, FactBits.TENPAI) != 0;
  }

  /** 判断の後、局の終わりまでにロンの役がある聴牌に届いたなら true を返す。和了も true。 */
  public boolean yakuTenpai(long[] words, int at) {
    return fact(words, at, FactBits.YAKU_TENPAI) != 0;
  }

  /** 自分の和了で受け取った点を返す。本場・供託は除き、和了しなければ 0。 */
  public int winPoints(long[] words, int at) {
    return fact(words, at, FactBits.WIN_POINTS) * ObservationSchema.POINT_UNIT;
  }

  /** 放銃で払った点の和を返す。本場は除き、放銃しなければ 0。 */
  public int dealInPoints(long[] words, int at) {
    return fact(words, at, FactBits.DEAL_IN_POINTS) * ObservationSchema.POINT_UNIT;
  }

  /** 和了の役のビットを返す。ビット k が {@link ScoringYaku} の ordinal k の役で、流局は 0。 */
  public long yakuBits(long[] words, int at) {
    return words[at + Word.YAKU.ordinal()];
  }

  /**
   * 局末の点を返す。
   *
   * @param words 記録を並べた配列
   * @param at 記録の先頭の位置
   * @param relative 判断した席から見た相対席（自分が 0）
   * @return 点
   */
  public int finalScore(long[] words, int at, int relative) {
    return scores.get(words, at + Word.SCORES.ordinal(), 0, ScoreBits.SCORE, relative) * ObservationSchema.POINT_UNIT;
  }

  /**
   * 局末の記録の形式のフィンガープリントを返す。行のマニフェストに書き、読むときに照合する。
   *
   * @return SHA-256 の先頭 8 B を 16 進で書いた文字列
   */
  public String fingerprint() {
    return fingerprint;
  }

  private int fact(long[] words, int at, FactBits field) {
    return facts.get(words, at + Word.FACTS.ordinal(), 0, field);
  }

  private void put(long[] words, int factsAt, FactBits field, int value) {
    facts.put(words, factsAt, 0, field, value);
  }

  /** 判断した席から見た相対席を、絶対席に直す。 */
  private int absolute(int seat, int relative) {
    return (seat + relative) % seats;
  }

  /** 絶対席を、判断した席から見た相対席に直す。 */
  private int relative(int seat, int player) {
    return Math.floorMod(player - seat, seats);
  }

  /** 点を {@link ObservationSchema#POINT_UNIT} 点単位の値に直す。 */
  private static int hundreds(int points) {
    if (points % ObservationSchema.POINT_UNIT != 0) {
      throw new IllegalStateException("Points are not a multiple of " + ObservationSchema.POINT_UNIT + ": " + points);
    }
    return points / ObservationSchema.POINT_UNIT;
  }

  /** 1 件の記録のワード。宣言の順に並べる。 */
  enum Word {
    /** 局末の事実（{@link FactBits}）。 */
    FACTS,
    /** 和了の役のビット。 */
    YAKU,
    /** 局末の点（{@link ScoreBits}）。 */
    SCORES
  }

  /** {@link Word#FACTS} のフィールド。 */
  enum FactBits implements BitField {
    /** 局の終わり方（{@link Outcome} の ordinal）。記録のない判断は −1。 */
    OUTCOME,
    /** 和了した他家の相対席 − 1。他家が和了していなければ −1。 */
    WINNER,
    /** 判断の後、局の終わりまでに聴牌に届いたら 1。和了も 1。 */
    TENPAI,
    /** 判断の後、局の終わりまでにロンの役がある聴牌に届いたら 1。和了も 1。 */
    YAKU_TENPAI,
    /** 自分の和了で受け取る点（本場・供託を除く）。 */
    WIN_POINTS,
    /** 放銃で払う点の和（本場を除く）。 */
    DEAL_IN_POINTS,
    /** 相手が局末に聴牌なら 1（和了を含む）。相手ごとに 1 つ持ち、下家から順に並べる（相対席 − 1 番目）。 */
    OPPONENT_TENPAI,
    /** 次の局の通し番号。半荘が終われば −1。 */
    NEXT_KYOKU,
    /** 次の局の本場。半荘が終われば 0。次の局の記録 O に入る値なので、上限は記録 O と同じ。 */
    NEXT_HONBA,
    /** 次の局に持ち越す供託の本数。半荘が終われば 0。 */
    NEXT_KYOTAKU;

    @Override
    public int min(Variant variant) {
      return switch (this) {
        case OUTCOME, WINNER, NEXT_KYOKU -> -1;
        default -> 0;
      };
    }

    @Override
    public int max(Variant variant) {
      return switch (this) {
        case OUTCOME -> Outcome.values().length - 1;
        case WINNER -> variant.seats() - 2;
        case TENPAI, YAKU_TENPAI, OPPONENT_TENPAI -> 1;
        case WIN_POINTS -> Limits.MAX_WIN_POINTS / ObservationSchema.POINT_UNIT;
        case DEAL_IN_POINTS -> Limits.MAX_DEAL_IN_POINTS / ObservationSchema.POINT_UNIT;
        case NEXT_KYOKU -> variant.finalRound();
        case NEXT_HONBA -> ObservationSchema.MAX_HONBA;
        case NEXT_KYOTAKU -> Limits.maxKyotaku(variant);
      };
    }

    @Override
    public int count(Variant variant) {
      return this == OPPONENT_TENPAI ? variant.seats() - 1 : 1;
    }
  }

  /** {@link Word#SCORES} のフィールド。 */
  enum ScoreBits implements BitField {
    /** 局末の点。{@link ObservationSchema#POINT_UNIT} 点単位の符号つき 16 bit。席ごとに 1 つ持ち、自分から相対席の順に並べる。 */
    SCORE;

    @Override
    public int min(Variant variant) {
      return Short.MIN_VALUE;
    }

    @Override
    public int max(Variant variant) {
      return Short.MAX_VALUE;
    }

    @Override
    public int count(Variant variant) {
      return variant.seats();
    }
  }
}
