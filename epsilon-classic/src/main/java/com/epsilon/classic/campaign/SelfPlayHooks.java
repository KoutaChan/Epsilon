package com.epsilon.classic.campaign;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.play.Run;
import com.epsilon.agent.play.RunHooks;
import com.epsilon.agent.play.TableHooks;
import com.epsilon.classic.learn.GameTrajectory;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.RankPermutations;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Resolution;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * classic の自己対局の卓の処理。半荘ごとに学習器の席（牌山の番号 mod 席の数）の軌跡（{@link GameTrajectory}）を書き、終えた半荘を
 * 受け取り先に渡す。半荘の ID は牌山の番号。
 *
 * <ul>
 *   <li>学習器の席の判断ごとに、判断の記録 O・選んだ手・卓がその手を引いた確率を書く。確率は席の打ち手が書いた分布 σ の σ(a)/Σσ（和が 0
 *       なら一様）で、学習器の席では振る舞いの方策 μ の値になる。合法手が 1 つの判断は O をここで取り込み、確率 1 で書く。
 *   <li>選び替えで局の進みが変わりえたかは、エンジンが手を進めた結果（{@link Run#resolution}）で決まる。それまで O を半荘に預け、次の判断の
 *       時点か局末で軌跡に書く。合法手が 1 つの判断と、他家のロンか優先する鳴きで結果が変わらなかった応答は false。
 *   <li>局の始まりごとに、GRP があればその卓の絶対席 × 順位の周辺確率を書く（無ければ書かない）。半荘の終わりに終局の点を書く。
 * </ul>
 *
 * <p>軌跡の中身は半荘の進みだけで決まり、実行の順やワーカーの数に依らない。
 */
final class SelfPlayHooks implements TableHooks<Query> {

  /** 預ける O のバッファの、最初の {@code long} の数。 */
  private static final int HELD_LONGS = 128;

  private final Variant variant;
  private final GrpNetwork grp;
  private final Consumer<GameTrajectory> sink;
  private final ThreadLocal<Tools> tools;

  /** 半荘の最初の局の周辺確率（GRP が無ければ null）。どの半荘も同じ卓から始まる。 */
  private final float[] first;

  /**
   * @param grp 局の始まりの周辺確率を出す GRP（null なら書かない）
   * @param sink 終えた半荘の受け取り先。進行のスレッドから並行に呼ぶ
   */
  SelfPlayHooks(Variant variant, GrpNetwork grp, Consumer<GameTrajectory> sink) {
    this.variant = Objects.requireNonNull(variant, "variant");
    if (grp != null && grp.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but the GRP is " + grp.variant());
    }
    this.grp = grp;
    this.sink = Objects.requireNonNull(sink, "sink");
    tools = ThreadLocal.withInitial(Tools::new);
    first = grp == null ? null : tools.get().prior(new TableState(variant));
  }

  /** 半荘の学習器の席（牌山の番号 mod 席の数）。 */
  static int learnerSeat(GameKey game, int seats) {
    return (int) Math.floorMod(game.wall(), (long) seats);
  }

  @Override
  public RunHooks<Query> main(Run<Query> run) {
    return new Game(run.game());
  }

  /** 進行のスレッドごとのヘルパー。どれもスレッドセーフでない。 */
  private final class Tools {
    final ObservationWriter writer = new ObservationWriter(variant);
    final Observation reader = new Observation(variant);
    final GrpWorkspace workspace = grp == null ? null : GrpWorkspace.of(grp.shape());
    final RankPermutations permutations = RankPermutations.of(variant);
    final double[] probabilities = new double[permutations.count()];
    final double[] marginals = new double[variant.seats() * variant.seats()];

    /** 半荘が続く卓の、絶対席 × 順位の周辺確率（新しい配列）。 */
    float[] prior(TableState start) {
      grp.permutations(start, workspace, probabilities);
      permutations.marginals(probabilities, start.dealer(), marginals);
      float[] prior = new float[marginals.length];
      for (int index = 0; index < prior.length; index++) {
        prior[index] = (float) marginals[index];
      }
      return prior;
    }
  }

  /** 1 半荘の主経路の処理。触るのは 1 度に 1 つのスレッドだけ。 */
  private final class Game implements RunHooks<Query> {

    private final GameTrajectory.Builder trajectory;
    private final int learner;

    /** 手を進めた結果を待つ学習器の判断の O（判断の時点に 1 つまで）。 */
    private final ObservationBatch held = new ObservationBatch(variant, HELD_LONGS);

    private boolean holding;
    private long heldId;
    private int heldChoice;
    private float heldProbability;

    Game(GameKey game) {
      trajectory = new GameTrajectory.Builder(variant, game.wall());
      learner = learnerSeat(game, variant.seats());
      trajectory.hand(first);
    }

    @Override
    public void boundary(Run<Query> run, Decisions decisions, RuleWorkspace ws) {
      flush(run);
    }

    @Override
    public Query query(
        Run<Query> run,
        Decisions decisions,
        int slot,
        SeatView view,
        int admission,
        ObservationBatch records,
        int record,
        Observation observation) {
      return new Query(run, run.game().wall(), slot, admission, records, record, observation);
    }

    @Override
    public void decided(Run<Query> run, Decisions decisions, int[] chosen, RuleWorkspace ws) {
      int slot = decisions.slotOf(learner);
      if (slot < 0) {
        return;
      }
      if (holding) {
        throw new IllegalStateException("Decision " + heldId + " of " + run + " was never resolved");
      }
      held.clear();
      Query query = run.query(slot);
      if (query == null) {
        tools.get().writer.capture(run.view(learner), decisions, slot, held);
        heldChoice = 0;
        heldProbability = 1f;
      } else {
        held.add(run.read(query));
        heldChoice = chosen[slot];
        heldProbability = drawn(run.distribution(slot), query.legalCount(), heldChoice);
      }
      heldId = decisions.id(slot);
      holding = true;
    }

    @Override
    public void roundEnded(Run<Query> run, RoundEnd end, boolean gameEnded) {
      flush(run);
      TableState next = end.next();
      if (!gameEnded) {
        trajectory.hand(grp == null ? null : tools.get().prior(next));
        return;
      }
      int[] scores = new int[variant.seats()];
      for (int seat = 0; seat < scores.length; seat++) {
        scores[seat] = next.score(seat);
      }
      sink.accept(trajectory.finish(scores));
    }

    /** 預けた判断を、エンジンが手を進めた結果と一緒に軌跡に書く。 */
    private void flush(Run<Query> run) {
      if (!holding) {
        return;
      }
      Resolution resolution = run.resolution();
      for (int slot = 0; slot < resolution.size(); slot++) {
        if (resolution.id(slot) == heldId) {
          boolean causal = resolution.role(slot) == Resolution.Role.CAUSAL;
          trajectory.decision(tools.get().reader.bind(held, 0), heldChoice, heldProbability, causal);
          holding = false;
          return;
        }
      }
      throw new IllegalStateException("The engine did not resolve decision " + heldId + " of " + run);
    }
  }

  /** 卓が分布 σ から手を引く確率 σ(a)/Σσ（和が 0 なら一様）。 */
  private static float drawn(float[] distribution, int legal, int choice) {
    double sum = 0;
    for (int action = 0; action < legal; action++) {
      sum += distribution[action];
    }
    return sum > 0 ? (float) (distribution[choice] / sum) : 1f / legal;
  }
}
