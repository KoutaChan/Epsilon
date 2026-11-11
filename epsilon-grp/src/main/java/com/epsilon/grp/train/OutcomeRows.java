package com.epsilon.grp.train;

import com.epsilon.grp.GrpKey;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.grp.WinMethod;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.stream.IntStream;

/**
 * OutcomeNetwork を学習する行を、値の種類ごとに 1 本の配列で持つ。1 行は半荘が続く局末 1 つで、その次の局の結果を学習する。
 * 行ごとに持つのは、局末の卓のキー（GrpKey）、範囲の最初のコホートからの差、結果ごとの対数確率の項の番号（HandOutcomes の項の
 * 番号）。項の番号は結果ごとに項の数の上限だけ並べ、空きは −1 にする。結果は、半荘の記録から作る行では記録した遷移を再現する結果の
 * 全て（TransitionSearch）、数のファイルから作る行では記録した結果 1 つ（OutcomeCodes）。行の損失は周辺尤度 −log Σ_o q(o|s) で、
 * 重みはコホートから引く（RecencyWindow）。行ごとのオブジェクトは作らない。学習用の行と早期終了用の検証の行は別の OutcomeRows に
 * 分け、半荘の分け方は GrpRows と同じ（CohortSplit）。
 */
public final class OutcomeRows implements TrainingLoop.Rows {

  private final Variant variant;
  private final CohortWeights weights;
  final long[] keys;
  final byte[] cohorts;

  /** 行ごとの最初の結果の番号。最後の要素は結果の総数。 */
  final int[] firstOutcome;

  /** 項の番号。配列の位置は（結果 × {@link #stride} ＋ 結果の中の何番目の項か）で、空きは −1。 */
  final short[] terms;

  /** 結果 1 つの項の数の上限（{@code HandOutcomes.maxTerms}）。 */
  final int stride;

  private final int size;
  private final int maxOutcomes;
  private final long excluded;
  private final long unexplained;

  private OutcomeRows(Variant variant, CohortWeights weights, Column column) {
    this.variant = variant;
    this.weights = weights;
    stride = column.stride;
    size = column.size;
    keys = Arrays.copyOf(column.keys, size);
    cohorts = Arrays.copyOf(column.cohorts, size);
    firstOutcome = new int[size + 1];
    int most = 0;
    for (int row = 0; row < size; row++) {
      firstOutcome[row + 1] = column.ends[row];
      most = Math.max(most, firstOutcome[row + 1] - firstOutcome[row]);
    }
    maxOutcomes = most;
    terms = Arrays.copyOf(column.terms, firstOutcome[size] * stride);
    excluded = column.excluded;
    unexplained = column.unexplained;
  }

  /**
   * 範囲に入るコホートの半荘から、学習用の行と早期終了用の検証の行を作るビルダーを返す。
   *
   * @param rows 行にする局末の卓を選ぶ条件（例えば {@code FinalRound::covers}）
   */
  public static Builder builder(Variant variant, RecencyWindow window, GateCohort gate, Predicate<GrpState> rows) {
    return new Builder(variant, window, gate, rows);
  }

  public Variant variant() {
    return variant;
  }

  public RecencyWindow window() {
    return weights.window();
  }

  @Override
  public int size() {
    return size;
  }

  public long key(int row) {
    return keys[Objects.checkIndex(row, size)];
  }

  public int cohort(int row) {
    return weights.cohort(cohorts[Objects.checkIndex(row, size)]);
  }

  public float weight(int row) {
    return weights.weight(cohorts[Objects.checkIndex(row, size)]);
  }

  /** 行の遷移を再現する結果の数（1 以上）。 */
  public int outcomes(int row) {
    Objects.checkIndex(row, size);
    return firstOutcome[row + 1] - firstOutcome[row];
  }

  /** 1 行の結果の数の最大。 */
  public int maxOutcomes() {
    return maxOutcomes;
  }

  /** 卓を GrpKey で表せないため外した局末の数。 */
  public long excluded() {
    return excluded;
  }

  /**
   * 局の結果のネットワークが表せる結果では遷移を再現できないため、外した局末の数。包や流し満貫のほか、ルールに合わない記録も含む
   * （どちらなのかは RuleCheck で分けられる）。数のファイルから作った行では、記録した結果を表せない局（包のある和了・流し満貫）の数。
   */
  public long unexplained() {
    return unexplained;
  }

  @Override
  public double totalWeight() {
    return weights.total(cohorts, size);
  }

  /** 純 Java のネットワークで、重み付きの周辺 NLL Σ w·(−log Σ_o q(o|s)) / Σ w を求める。 */
  public double nll(OutcomeNetwork network, OutcomeWorkspace workspace) {
    if (network.variant() != variant) {
      throw new IllegalArgumentException("A " + network.variant() + " network for " + variant + " rows");
    }
    double[] logs = new double[maxOutcomes];
    double sum = 0;
    for (int row = 0; row < size; row++) {
      network.forward(GrpState.ofKey(variant, keys[row]), workspace);
      sum -= weightOf(row) * logLikelihood(row, workspace, logs);
    }
    return sum / totalWeight();
  }

  /**
   * 行の log Σ_o q(o|s) を返す。卓の確率は作業領域に最後に求めたものを使うので、その行の卓で順伝播した後に呼ぶ。
   *
   * @param logs 結果ごとの対数確率を書く作業用の配列（長さは 1 行の結果の数の最大以上）
   */
  double logLikelihood(int row, OutcomeWorkspace workspace, double[] logs) {
    int count = firstOutcome[row + 1] - firstOutcome[row];
    double max = Double.NEGATIVE_INFINITY;
    for (int k = 0; k < count; k++) {
      double log = 0;
      int at = (firstOutcome[row] + k) * stride;
      for (int j = 0; j < stride && terms[at + j] >= 0; j++) {
        log += workspace.term(terms[at + j]);
      }
      logs[k] = log;
      max = Math.max(max, log);
    }
    double total = 0;
    for (int k = 0; k < count; k++) {
      total += StrictMath.exp(logs[k] - max);
    }
    return max + StrictMath.log(total);
  }

  /** 行の重み。学習のループの中で行ごとに呼ぶので、配列の位置は確かめない。 */
  float weightOf(int row) {
    return weights.weight(cohorts[row]);
  }

  /**
   * 結果 1 つについて、対数確率の項の番号を書く。順は終わり方、和了ごとの打点、席ごとの立直の有無で、立直の有無は確率に入れる席の
   * 分だけ書く。
   *
   * @param main 終わり方の番号
   * @param base 和了者（ダブロンなら {@code HandOutcomes.winner} の席）の打点の番号。和了でなければ読まない
   * @param other ダブロンの 2 人目の打点の番号。ダブロンでなければ読まない
   * @param riichi 立直が成立した席のビット（相対席）
   * @param out 書き先
   * @param at 書き始める位置
   * @return 書いた項の数
   */
  static int outcomeTerms(HandOutcomes hands, int main, int base, int other, int riichi, short[] out, int at) {
    int next = at;
    out[next++] = (short) hands.mainTerm(main);
    switch (hands.section(main)) {
      case TSUMO -> out[next++] = (short) hands.gradeTerm(hands.winner(main), WinMethod.TSUMO, base);
      case RON -> out[next++] = (short) hands.gradeTerm(hands.winner(main), WinMethod.RON, base);
      case DOUBLE_RON -> {
        out[next++] = (short) hands.gradeTerm(hands.winner(main), WinMethod.RON, base);
        out[next++] = (short) hands.gradeTerm(hands.secondWinner(main), WinMethod.RON, other);
      }
      case DRAW, ABORTIVE -> {
        // 打点の項は無い。
      }
    }
    for (int factor = 0; factor < hands.factorCount(main); factor++) {
      int seat = hands.factorSeat(main, factor);
      out[next++] = (short) hands.riichiTerm(seat, hands.role(main, seat), (riichi & 1 << seat) != 0);
    }
    return next - at;
  }

  @Override
  public String toString() {
    return "OutcomeRows[" + variant + " " + window() + " rows=" + size + " outcomes=" + firstOutcome[size]
        + " excluded=" + excluded + " unexplained=" + unexplained + "]";
  }

  /** 範囲の学習用の行と、早期終了用の検証の行。 */
  public record Split(OutcomeRows train, OutcomeRows stop) {}

  /**
   * 範囲のコホートを 1 つずつ足して行を作る。半荘の記録から足すときは、遷移を半荘ごとに並列に探し、見つけた行は半荘の順に足す。
   * そのため行の並びは並列の進み方に依らない。数のファイルから足すときは、半荘 ID の順、局の順に足す。
   */
  public static final class Builder {

    private final Variant variant;
    private final Predicate<GrpState> rows;
    private final HandOutcomes hands;
    private final CohortSplit split;
    private final Column train;
    private final Column stop;
    private final ThreadLocal<TransitionSearch> searches;
    private final OutcomeCodes codes;

    private Builder(Variant variant, RecencyWindow window, GateCohort gate, Predicate<GrpState> rows) {
      this.variant = variant;
      this.rows = Objects.requireNonNull(rows, "rows");
      split = new CohortSplit(variant, window, gate);
      hands = HandOutcomes.of(variant);
      train = new Column(hands.maxTerms());
      stop = new Column(hands.maxTerms());
      searches = ThreadLocal.withInitial(() -> new TransitionSearch(variant));
      codes = new OutcomeCodes(variant);
    }

    /** コホート 1 つの半荘から行を足す。範囲の外のコホート、2 度目に足すコホート、ルールの違う半荘には IllegalArgumentException を投げる。 */
    public Builder add(int cohort, List<GameRecord> games) {
      CohortSplit.Target[] targets = split.accept(cohort, games);
      byte offset = split.weights().offset(cohort);
      List<Column> found = IntStream.range(0, games.size())
          .parallel()
          .mapToObj(index -> targets[index] == CohortSplit.Target.NONE ? null : search(games.get(index)))
          .toList();
      for (int index = 0; index < games.size(); index++) {
        switch (targets[index]) {
          case TRAIN -> train.append(found.get(index), offset);
          case STOP -> stop.append(found.get(index), offset);
          case NONE -> {
            // 行にしない。
          }
        }
      }
      return this;
    }

    /**
     * コホート 1 つの数のファイル（CountedHands）から行を足す。条件を満たす局の始まりの卓ごとに、記録した結果 1 つだけの行を作る。
     *
     * @throws IllegalArgumentException 範囲の外のコホート、2 度目に足すコホート、ルールの違う数のファイルのとき
     */
    public Builder add(int cohort, CountedHands counted) {
      if (counted.variant() != variant) {
        throw new IllegalArgumentException("A " + counted.variant() + " count file in " + variant + " rows");
      }
      CohortSplit.Target[] targets = split.accept(cohort, counted.games(), counted::gameId);
      byte offset = split.weights().offset(cohort);
      int[] scores = new int[variant.seats()];
      for (int game = 0; game < counted.games(); game++) {
        Column column = switch (targets[game]) {
          case TRAIN -> train;
          case STOP -> stop;
          case NONE -> null;
        };
        if (column == null) {
          continue;
        }
        for (int hand = counted.firstHand(game); hand < counted.firstHand(game + 1); hand++) {
          for (int seat = 0; seat < scores.length; seat++) {
            scores[seat] = counted.score(hand, seat);
          }
          GrpState start =
              new GrpState(variant, counted.round(hand), counted.honba(hand), counted.kyotaku(hand), scores);
          if (!rows.test(start)) {
            continue;
          }
          long key = start.key();
          if (key == GrpKey.NONE) {
            column.excluded++;
          } else if (!column.add(key, counted.code(hand), codes, offset)) {
            column.unexplained++;
          }
        }
      }
      return this;
    }

    public Split build() {
      return new Split(
          new OutcomeRows(variant, split.weights(), train), new OutcomeRows(variant, split.weights(), stop));
    }

    /** 半荘 1 つの行を作る。条件を満たす局末ごとに、その次の局への遷移を再現する結果を探す。 */
    private Column search(GameRecord game) {
      TransitionSearch search = searches.get();
      Column column = new Column(hands.maxTerms());
      List<GrpState> handEnds = game.handEnds();
      for (int handEnd = 0; handEnd < handEnds.size(); handEnd++) {
        GrpState state = handEnds.get(handEnd);
        if (!rows.test(state)) {
          continue;
        }
        long key = state.key();
        if (key == GrpKey.NONE) {
          column.excluded++;
          continue;
        }
        search.from(state);
        int found = handEnd + 1 < handEnds.size()
            ? search.next(handEnds.get(handEnd + 1))
            : search.last(game.finalScores());
        if (found == 0) {
          column.unexplained++;
        } else {
          column.add(key, search, hands);
        }
      }
      return column;
    }
  }

  /** 作っている途中の行。足りなくなれば配列を倍に伸ばし、OutcomeRows にコピーするときに切り詰める。 */
  private static final class Column {

    private final int stride;
    private long[] keys = new long[64];
    private byte[] cohorts = new byte[keys.length];

    /** 行ごとに、その行までの結果の累計。 */
    private int[] ends = new int[keys.length];

    private short[] terms;
    private int size;
    private int outcomes;
    private long excluded;
    private long unexplained;

    Column(int stride) {
      this.stride = stride;
      terms = new short[keys.length * stride];
    }

    /** 探索で見つけた結果を 1 行にする。行のコホートはここでは書かず、{@code append} で書く。 */
    void add(long key, TransitionSearch search, HandOutcomes hands) {
      growRows();
      int found = search.found();
      while (terms.length < (outcomes + found) * stride) {
        terms = Arrays.copyOf(terms, 2 * terms.length);
      }
      for (int index = 0; index < found; index++) {
        write(hands, search, index, (outcomes + index) * stride);
      }
      outcomes += found;
      keys[size] = key;
      ends[size] = outcomes;
      size++;
    }

    /**
     * 記録した結果 1 つの行を、指定のコホートの行として足す。
     *
     * @param code 数のファイルにある、局の結果のコード
     * @return 足したら true。局の結果のネットワークが表せない結果なら false で、何も足さない
     */
    boolean add(long key, long code, OutcomeCodes codes, byte cohort) {
      while (terms.length < (outcomes + 1) * stride) {
        terms = Arrays.copyOf(terms, 2 * terms.length);
      }
      int at = outcomes * stride;
      int written = codes.terms(code, terms, at);
      if (written == 0) {
        return false;
      }
      Arrays.fill(terms, at + written, at + stride, (short) -1);
      growRows();
      outcomes++;
      keys[size] = key;
      cohorts[size] = cohort;
      ends[size] = outcomes;
      size++;
      return true;
    }

    /** 半荘 1 つの行を、指定のコホートの行として足す。 */
    void append(Column game, byte cohort) {
      for (int row = 0; row < game.size; row++) {
        growRows();
        keys[size] = game.keys[row];
        cohorts[size] = cohort;
        ends[size] = outcomes + game.ends[row];
        size++;
      }
      int added = game.outcomes * stride;
      while (terms.length < outcomes * stride + added) {
        terms = Arrays.copyOf(terms, 2 * terms.length);
      }
      System.arraycopy(game.terms, 0, terms, outcomes * stride, added);
      outcomes += game.outcomes;
      excluded += game.excluded;
      unexplained += game.unexplained;
    }

    private void growRows() {
      if (size == keys.length) {
        int capacity = 2 * keys.length;
        keys = Arrays.copyOf(keys, capacity);
        cohorts = Arrays.copyOf(cohorts, capacity);
        ends = Arrays.copyOf(ends, capacity);
      }
    }

    /** 見つけた結果 1 つについて、項の番号を指定の位置から書き、結果 1 つぶんの残りを −1 で埋める。 */
    private void write(HandOutcomes hands, TransitionSearch search, int index, int at) {
      int written = outcomeTerms(
          hands, search.main(index), search.base(index), search.other(index), search.riichi(index), terms, at);
      Arrays.fill(terms, at + written, at + stride, (short) -1);
    }
  }
}
