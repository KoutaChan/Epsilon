package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.PolicySeat;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TablePlan;
import com.epsilon.agent.play.TableSeat;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.ClassicFamily;
import com.epsilon.classic.learn.Exploration;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.learn.UncertaintyCdf;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 自己対局の macro の収集。学習器を半荘ごとに 1 席（牌山の番号 mod 席の数）に、macro の相手を残りの全ての席に置いて卓の
 * コアで半荘を打ち、学習器の席の軌跡を集める（{@link SelfPlayHooks}）。学習器は振る舞いの方策から手を引き（{@link LearnerSeat}）、
 * 相手は貪欲に打つ。
 *
 * <p>牌山・局のキー・相手の選び・迷いの百分位を散らす一様値は、run の種と macro の番号だけで決まる。牌山の番号は macro ごとに 0 から振る。
 */
public final class Collect {

  private static final Logger LOG = LoggerFactory.getLogger(Collect.class);

  /** run の種に混ぜるソルト（牌山・局のキー・相手の選び・迷いの百分位を散らす一様値）。 */
  private static final long WALLS = 0x434c_5741_4c4c_5331L, HANDS = 0x434c_4841_4e44_5331L,
      OPPONENT = 0x434c_4f50_504f_4e54L, TIES = 0x434c_5449_4553_5331L;

  /**
   * 収集の設定。
   *
   * @param games 1 回の macro の半荘の数（席の数の倍数なら、学習器はどの席にも同じ数だけ座る）
   * @param championProbability 相手に今の王者を選ぶ確率（残りは過去の王者から一様に 1 人）
   */
  public record Settings(int games, double championProbability) {

    public Settings {
      if (games < 1) {
        throw new IllegalArgumentException("A macro needs at least one game: " + games);
      }
      if (!(championProbability >= 0 && championProbability <= 1)) {
        throw new IllegalArgumentException("The champion probability must be in [0, 1]: " + championProbability);
      }
    }
  }

  /**
   * 1 回の macro の収集の結果。
   *
   * @param trajectories 学習器の席の軌跡（半荘 ID の順）
   * @param calibration この macro で数えた迷いの分布（次の macro の探索が引く）
   * @param opponent 相手
   */
  public record Result(TrajectoryBatch trajectories, UncertaintyCdf calibration, Champion opponent) {}

  /** 打った半荘の軌跡と、卓のコアの結果。 */
  record Played(TrajectoryBatch trajectories, TableLoop.Result table) {}

  private final Variant variant;
  private final long runSeed;
  private final Settings settings;
  private final Exploration.Settings exploration;
  private final GrpNetwork grp;
  private final PolicyOptions options;
  private final NetworkRuntime runtime;

  /**
   * @param runSeed run の種
   * @param exploration 学習器の探索の設定
   * @param grp 局の始まりの周辺確率を出す GRP（null なら軌跡に書かない）
   * @param options 学習器と相手の方策を開く設定
   * @param runtime GPU の方策が使う推論の実行環境（null なら方策が自分で持つ。閉じるのは呼び出し元）
   */
  public Collect(
      Variant variant,
      long runSeed,
      Settings settings,
      Exploration.Settings exploration,
      GrpNetwork grp,
      PolicyOptions options,
      NetworkRuntime runtime) {
    this.variant = Objects.requireNonNull(variant, "variant");
    this.runSeed = runSeed;
    this.settings = Objects.requireNonNull(settings, "settings");
    this.exploration = Objects.requireNonNull(exploration, "exploration");
    if (grp != null && grp.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but the GRP is " + grp.variant());
    }
    this.grp = grp;
    this.options = Objects.requireNonNull(options, "options");
    this.runtime = runtime;
  }

  /**
   * 1 回の macro を集める。相手を選び、学習器と相手の方策を開いて半荘を打ち、方策を閉じる。相手は学習器と同じアーキテクチャのチェックポイント。
   * 持ち主が置いた初めの王者を相手に選んだら、開く前に、置いたときの SHA-256 と照合する。
   *
   * @param macro macro の番号（0 以上）
   * @param learner 学習器のチェックポイント
   * @param opponents 相手の候補
   * @param previous 前の macro の迷いの分布（初めの macro は {@link UncertaintyCdf#empty}）
   * @throws IllegalStateException チェックポイントのアーキテクチャ・ルール・入力のフィンガープリントが合わない
   * @throws IOException チェックポイントが読めないか、置いた初めの王者のチェックポイントが置いたときの中身と違う
   */
  public Result macro(TableLoop table, long macro, Path learner, OpponentPool opponents, UncertaintyCdf previous)
      throws IOException {
    if (macro < 0) {
      throw new IllegalArgumentException("The macro must not be negative: " + macro);
    }
    Champion opponent = opponents.draw(settings.championProbability(), SeedMixer.derive(runSeed, OPPONENT, macro));
    if (DuelJobs.placed(opponent)) {
      opponent.verify();
    }
    LOG.info("classic collect macro {} learner={} opponent={} ({})", macro, learner, opponent.label(), opponent.checkpoint());
    Architecture architecture = ClassicCheckpoint.header(learner).architecture();
    Exploration explore = new Exploration(exploration, previous);
    Played played;
    try (LearnerSeat learnerSeat =
            new LearnerSeat(
                ClassicFamily.open(architecture, learner, variant, options, runtime),
                explore,
                SeedMixer.derive(runSeed, TIES, macro));
        PolicySeat opponentSeat =
            new PolicySeat(
                ClassicFamily.open(architecture, opponent.checkpoint(), variant, options, runtime).asPolicy())) {
      played = play(table, macro, learnerSeat, opponentSeat, null);
    }
    return new Result(played.trajectories(), explore.calibration(), opponent);
  }

  /**
   * 学習器と相手の席で macro の半荘を打ち、学習器の席の軌跡を集める。席は閉じない。
   *
   * @param observer 主経路の観測（null なら観測しない）
   */
  Played play(TableLoop table, long macro, TableSeat<Query> learner, TableSeat<Query> opponent, GameObserver observer) {
    int seats = variant.seats();
    List<List<TableSeat<Query>>> seatings = new ArrayList<>(seats);
    for (int learnerSeat = 0; learnerSeat < seats; learnerSeat++) {
      List<TableSeat<Query>> seating = new ArrayList<>(seats);
      for (int seat = 0; seat < seats; seat++) {
        seating.add(seat == learnerSeat ? learner : opponent);
      }
      seatings.add(List.copyOf(seating));
    }
    List<GameKey> games = new ArrayList<>(settings.games());
    for (int wall = 0; wall < settings.games(); wall++) {
      games.add(new GameKey(wall, 0));
    }
    TrajectoryBatch.Collector collector = new TrajectoryBatch.Collector(variant);
    TablePlan<Query> plan =
        new TablePlan<>(
            variant,
            SeedMixer.derive(runSeed, WALLS, macro),
            SeedMixer.derive(runSeed, HANDS, macro),
            games,
            DrawOrder.SEAT_STREAMS,
            null,
            game -> seatings.get(SelfPlayHooks.learnerSeat(game, seats)),
            observer);
    TableLoop.Result result = table.play(plan, new SelfPlayHooks(variant, grp, collector::add));
    TrajectoryBatch trajectories = collector.build();
    if (trajectories.games().size() != settings.games()) {
      throw new IllegalStateException(
          "Macro " + macro + " collected " + trajectories.games().size() + " of " + settings.games() + " games");
    }
    LOG.info(
        "classic collect macro {} games={} decisions={} rounds={} queried={} forced={} seconds={}",
        macro, trajectories.games().size(), trajectories.size(), result.rounds(), result.queriedDecisions(),
        result.forcedDecisions(), result.seconds());
    return new Played(trajectories, result);
  }
}
