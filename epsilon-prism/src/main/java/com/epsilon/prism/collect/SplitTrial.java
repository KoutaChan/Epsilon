package com.epsilon.prism.collect;

import com.epsilon.agent.play.TableLoop;
import com.epsilon.grp.Phi;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.eval.GameEstimate;
import com.epsilon.prism.luck.LedgerCorrection;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

/** 主経路で選択が割れた判断だけ二つに分岐する。同じ半荘の分岐は判断順に集計する。 */
public final class SplitTrial {
  public record Result(String group, GameEstimate all, GameEstimate disagreements) {}

  private static final class Game {
    final GameEstimate.Game all = new GameEstimate.Game();
    final GameEstimate.Game disagreements = new GameEstimate.Game();
    final ArrayDeque<Pair> pending = new ArrayDeque<>();

    synchronized void agree(DecisionClass group) {
      all.add(group, 0);
    }

    synchronized Pair start(DecisionClass group, int seat, int a, double[] beta) {
      var pair = new Pair(this, group, seat, a, beta);
      pending.addLast(pair);
      return pair;
    }

    /** 局末が両方そろった先頭の判断までを、主経路での発生順に取り込む。 */
    synchronized void settle() {
      while (!pending.isEmpty() && pending.getFirst().complete()) {
        Pair pair = pending.removeFirst();
        double difference = pair.first - pair.second;
        all.add(pair.group, difference);
        disagreements.add(pair.group, difference);
      }
    }
  }

  private final TreeMap<Long, Game> games = new TreeMap<>();
  final PolicyService first;
  final PolicyService second;
  private final double rate;
  private final LedgerCorrection correction;

  private SplitTrial(
      Collector.Cohort cohort,
      double rate,
      PolicyService first,
      PolicyService second,
      LedgerCorrection correction) {
    if (!(rate > 0 && rate <= 1)) throw new IllegalArgumentException("q must be in (0, 1]");
    if ((cohort.ledger() == null) != (correction == null))
      throw new IllegalArgumentException("Incomplete ledger");
    this.first = first;
    this.second = second;
    this.rate = rate;
    this.correction = correction;
    for (long id : cohort.gameIds()) games.put(id, new Game());
  }

  public static List<Result> play(
      TableLoop table,
      Collector.Cohort cohort,
      PolicyService mu,
      Phi phi,
      double q,
      PolicyService a,
      PolicyService b,
      LedgerCorrection correction) {
    var trial = new SplitTrial(cohort, q, a, b, correction);
    Collector.play(table, cohort, mu, phi, null, trial);
    return trial.results();
  }

  boolean sampled(long key, long decision) {
    return CrnSeeds.coin(key, CrnSeeds.Domain.LABEL, decision) < rate;
  }

  void agree(long wall, DecisionClass group) {
    games.get(wall).agree(group);
  }

  Pair disagree(PolicyRequest request, int a) {
    DecisionClass group = request.decisionClass();
    double[] beta = null;
    if (correction != null) {
      int cell =
          correction.cells() == null
              ? 0
              : correction.cells().of(request.recordWords(), request.recordBase());
      beta = correction.coefficients(request.wall(), group.ordinal(), cell);
    }
    return games.get(request.wall()).start(group, request.seat(), a, beta);
  }

  private List<Result> results() {
    for (Game game : games.values()) {
      if (!game.pending.isEmpty()) throw new IllegalStateException("Unsettled split decisions");
    }
    var all = games.values().stream().map(g -> g.all).toList();
    var different = games.values().stream().map(g -> g.disagreements).toList();
    var results = new ArrayList<Result>();
    for (DecisionClass group : DecisionClass.values()) {
      results.add(
          new Result(group.name(), GameEstimate.of(all, group), GameEstimate.of(different, group)));
    }
    results.add(new Result("ALL", GameEstimate.of(all, null), GameEstimate.of(different, null)));
    return results;
  }

  static final class Pair implements Pending {
    private final Game game;
    private final DecisionClass group;
    private final int seat;
    private final int a;
    private final double[] beta;
    private double first = Double.NaN, second = Double.NaN, adjustment;

    Pair(Game game, DecisionClass group, int seat, int a, double[] beta) {
      this.game = game;
      this.group = group;
      this.seat = seat;
      this.a = a;
      this.beta = beta;
    }

    @Override
    public int seat() {
      return seat;
    }

    @Override
    public DecisionClass decisionClass() {
      return group;
    }

    @Override
    public void acceptLedger(int action, double[] terms) {
      adjustment = 0;
      for (int i = 0; i < terms.length; i++) adjustment += beta[i] * terms[i];
    }

    @Override
    public void accept(int action, double[] values) {
      if (action == a) first = values[seat] - adjustment;
      else second = values[seat] - adjustment;
      game.settle();
    }

    private boolean complete() {
      return !Double.isNaN(first) && !Double.isNaN(second);
    }
  }
}
