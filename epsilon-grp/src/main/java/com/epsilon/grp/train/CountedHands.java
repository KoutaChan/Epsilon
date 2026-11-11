package com.epsilon.grp.train;

import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Objects;

/**
 * コホート 1 つについて、主経路で数えた局の列。数のファイルの中身にあたる。半荘は半荘 ID の昇順に並び、半荘の中の局は打った順に
 * 並ぶ。局ごとに、その局の始まりの卓と、結果のコード（HandResult の生の形）を持つ。半荘が終わった局も入る。ほかに、主経路の探索の
 * 温度（正なら、主経路は方策のままでは打っていない）と、主経路の局末の値に使った Φ を持つ。作った後は書き換えない。
 */
public final class CountedHands {

  /** 局の番号の上限と、本場・供託の本数の上限。数のファイルの u8・u16 のフィールドに入る最大の値。 */
  static final int MAX_ROUND = 0xFF;

  static final int MAX_COUNT = 0xFFFF;

  private final Variant variant;
  private final int cohort;
  private final double exploreTemperature;
  private final PhiStamp phi;
  private final long[] gameIds;

  /** 半荘ごとの最初の局。最後の要素は局の数。 */
  private final int[] firstHand;

  private final int[] rounds;
  private final int[] honba;
  private final int[] kyotaku;

  /** 局の始まりの点。並びは [局 × S ＋ 絶対席]。 */
  private final int[] scores;

  private final long[] codes;

  private CountedHands(
      Variant variant, int cohort, double exploreTemperature, PhiStamp phi, long[] gameIds, int[] firstHand,
      int[] rounds, int[] honba, int[] kyotaku, int[] scores, long[] codes) {
    this.variant = variant;
    this.cohort = cohort;
    this.exploreTemperature = exploreTemperature;
    this.phi = phi;
    this.gameIds = gameIds;
    this.firstHand = firstHand;
    this.rounds = rounds;
    this.honba = honba;
    this.kyotaku = kyotaku;
    this.scores = scores;
    this.codes = codes;
  }

  /**
   * 局を足していく。
   *
   * @param exploreTemperature 主経路の探索の温度（0 以上の有限の数）
   * @param phi 主経路の局末の値に使った Φ
   */
  public static Builder builder(Variant variant, int cohort, double exploreTemperature, PhiStamp phi) {
    return new Builder(variant, cohort, exploreTemperature, phi);
  }

  public Variant variant() {
    return variant;
  }

  public int cohort() {
    return cohort;
  }

  /** 主経路の探索の温度。0 なら、主経路は方策のままで打った。 */
  public double exploreTemperature() {
    return exploreTemperature;
  }

  /** 主経路の局末の値に使った Φ。 */
  public PhiStamp phi() {
    return phi;
  }

  public int games() {
    return gameIds.length;
  }

  public long gameId(int game) {
    return gameIds[game];
  }

  /** 半荘の最初の局。半荘 g の局は {@code firstHand(g)} から {@code firstHand(g + 1)} の手前まで。 */
  public int firstHand(int game) {
    return firstHand[game];
  }

  /** 局の数。 */
  public int hands() {
    return codes.length;
  }

  public int round(int hand) {
    return rounds[hand];
  }

  public int honba(int hand) {
    return honba[hand];
  }

  public int kyotaku(int hand) {
    return kyotaku[hand];
  }

  /** 局の始まりの点。 */
  public int score(int hand, int seat) {
    Objects.checkIndex(seat, variant.seats());
    return scores[hand * variant.seats() + seat];
  }

  /** 局の結果のコード。 */
  public long code(int hand) {
    return codes[hand];
  }

  /**
   * 局の始まりの卓を、渡した卓に写す。点の和は、局を足したときに確かめてある。
   *
   * @param into 写す先（このルールの卓）
   */
  public void start(int hand, TableState into) {
    int seats = variant.seats();
    into.set(rounds[hand], honba[hand], kyotaku[hand], Arrays.copyOfRange(scores, hand * seats, (hand + 1) * seats));
  }

  /**
   * 局ごとに、始まりの卓から結果どおりに精算して次の局へ進め、その卓が同じ半荘の次の局の始まりの卓と同じか（半荘の最後の局なら
   * 終局か）を確かめる。
   *
   * @return 外れた最初の局の番号。全て合えば −1
   */
  public int firstMismatch() {
    TableState table = new TableState(variant);
    TableState next = new TableState(variant);
    RoundOutcome outcome = new RoundOutcome(variant);
    int[] delta = new int[variant.seats()];
    for (int game = 0; game < games(); game++) {
      for (int hand = firstHand[game]; hand < firstHand[game + 1]; hand++) {
        start(hand, table);
        HandResult.place(codes[hand], table.dealer(), outcome);
        Settlement.apply(variant, table, outcome, delta);
        Progression.next(table, outcome, delta, next);
        boolean last = hand == firstHand[game + 1] - 1;
        if (last ? !next.gameOver() : next.gameOver() || !startsAt(hand + 1, next)) {
          return hand;
        }
      }
    }
    return -1;
  }

  /** 卓が、その局の始まりの卓と同じか。 */
  private boolean startsAt(int hand, TableState table) {
    if (table.round() != rounds[hand] || table.honba() != honba[hand] || table.kyotaku() != kyotaku[hand]) {
      return false;
    }
    int seats = variant.seats();
    for (int seat = 0; seat < seats; seat++) {
      if (table.score(seat) != scores[hand * seats + seat]) {
        return false;
      }
    }
    return true;
  }

  @Override
  public String toString() {
    return "CountedHands[" + variant + " cohort=" + cohort + " games=" + games() + " hands=" + hands()
        + " exploreTemperature=" + exploreTemperature + " phi=" + phi + "]";
  }

  /**
   * 局を足して数の列を作る。半荘の局は打った順に続けて足す（半荘 ID の同じ局が続く間は同じ半荘とみなす）。半荘を足す順は問わず、
   * {@code build} が半荘 ID の昇順に並べ替える。同時に使えるのは 1 スレッドだけ。
   */
  public static final class Builder {

    private static final int INITIAL = 1 << 10;

    private final Variant variant;
    private final int cohort;
    private final double exploreTemperature;
    private final PhiStamp phi;
    private final int seats;
    private final RoundOutcome scratch;
    private long[] gameIds = new long[INITIAL];
    private int[] gameFirst = new int[INITIAL];
    private int games;
    private int[] rounds = new int[INITIAL];
    private int[] honba = new int[INITIAL];
    private int[] kyotaku = new int[INITIAL];
    private int[] scores;
    private long[] codes = new long[INITIAL];
    private int hands;
    private boolean built;

    private Builder(Variant variant, int cohort, double exploreTemperature, PhiStamp phi) {
      this.variant = Objects.requireNonNull(variant, "variant");
      if (!(exploreTemperature >= 0 && Double.isFinite(exploreTemperature))) {
        throw new IllegalArgumentException("The explore temperature must be finite and not negative: "
            + exploreTemperature);
      }
      this.cohort = cohort;
      this.exploreTemperature = exploreTemperature;
      this.phi = Objects.requireNonNull(phi, "phi");
      seats = variant.seats();
      scratch = new RoundOutcome(variant);
      scores = new int[INITIAL * seats];
    }

    /**
     * 局を 1 つ足す。直前に足した局と半荘 ID が違えば、新しい半荘を始める。
     *
     * @param gameId 半荘 ID
     * @param start 局の始まりの卓。終局していない卓で、読むだけ
     * @param code その局の結果のコード（{@link HandResult#valid} を満たすこと）
     * @throws IllegalArgumentException ルールが違うとき、終局の卓のとき、点と供託の和が配給点の和と違うとき、コードが決まりを
     *     破るとき、局・本場・供託が数のファイルのフィールドに入らないとき
     */
    public Builder add(long gameId, TableState start, long code) {
      if (built) {
        throw new IllegalStateException("The counts were already built");
      }
      if (start.variant() != variant) {
        throw new IllegalArgumentException("Variant mismatch: counts " + variant + ", table " + start.variant());
      }
      if (start.gameOver()) {
        throw new IllegalArgumentException("A finished game starts no hand: " + start);
      }
      if (start.round() > MAX_ROUND || start.honba() > MAX_COUNT || start.kyotaku() > MAX_COUNT) {
        throw new IllegalArgumentException("The table does not fit the count file: " + start);
      }
      long total = (long) Variant.DEPOSIT * start.kyotaku();
      for (int seat = 0; seat < seats; seat++) {
        total += start.score(seat);
      }
      if (total != (long) seats * variant.startPoints()) {
        throw new IllegalArgumentException("Scores and deposits do not add up to the starting total: " + start);
      }
      if (!HandResult.valid(code, scratch)) {
        throw new IllegalArgumentException("Not a " + variant + " hand result: " + Long.toHexString(code));
      }
      if (games == 0 || gameIds[games - 1] != gameId) {
        if (games == gameIds.length) {
          gameIds = Arrays.copyOf(gameIds, 2 * games);
          gameFirst = Arrays.copyOf(gameFirst, 2 * games);
        }
        gameIds[games] = gameId;
        gameFirst[games] = hands;
        games++;
      }
      if (hands == codes.length) {
        rounds = Arrays.copyOf(rounds, 2 * hands);
        honba = Arrays.copyOf(honba, 2 * hands);
        kyotaku = Arrays.copyOf(kyotaku, 2 * hands);
        scores = Arrays.copyOf(scores, 2 * hands * seats);
        codes = Arrays.copyOf(codes, 2 * hands);
      }
      rounds[hands] = start.round();
      honba[hands] = start.honba();
      kyotaku[hands] = start.kyotaku();
      for (int seat = 0; seat < seats; seat++) {
        scores[hands * seats + seat] = start.score(seat);
      }
      codes[hands] = code;
      hands++;
      return this;
    }

    /**
     * 半荘を半荘 ID の昇順に並べた数の列を作る。
     *
     * @throws IllegalStateException 同じ半荘 ID の局が離れて足されたとき（半荘が 2 度現れるとき）、半荘の局の数が数のファイルに
     *     入らないとき、2 度目の呼び出しのとき
     */
    public CountedHands build() {
      if (built) {
        throw new IllegalStateException("The counts were already built");
      }
      built = true;
      long[] ids = Arrays.copyOf(gameIds, games);
      LongIndex order;
      try {
        order = LongIndex.of(ids);
      } catch (IllegalArgumentException repeated) {
        throw new IllegalStateException("A game was added in two separate runs: " + repeated.getMessage(), repeated);
      }
      long[] sorted = ids.clone();
      Arrays.sort(sorted);
      int[] first = new int[games + 1];
      int[] sortedRounds = new int[hands];
      int[] sortedHonba = new int[hands];
      int[] sortedKyotaku = new int[hands];
      int[] sortedScores = new int[hands * seats];
      long[] sortedCodes = new long[hands];
      int at = 0;
      for (int game = 0; game < games; game++) {
        int source = order.get(sorted[game]);
        int from = gameFirst[source];
        int to = source + 1 < games ? gameFirst[source + 1] : hands;
        first[game] = at;
        int count = to - from;
        if (count > MAX_COUNT) {
          throw new IllegalStateException("Game " + sorted[game] + " has more hands than the count file holds: " + count);
        }
        System.arraycopy(rounds, from, sortedRounds, at, count);
        System.arraycopy(honba, from, sortedHonba, at, count);
        System.arraycopy(kyotaku, from, sortedKyotaku, at, count);
        System.arraycopy(scores, from * seats, sortedScores, at * seats, count * seats);
        System.arraycopy(codes, from, sortedCodes, at, count);
        at += count;
      }
      first[games] = at;
      return new CountedHands(
          variant, cohort, exploreTemperature, phi, sorted, first, sortedRounds, sortedHonba, sortedKyotaku,
          sortedScores, sortedCodes);
    }
  }
}
