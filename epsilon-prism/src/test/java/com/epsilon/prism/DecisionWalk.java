package com.epsilon.prism;

import com.epsilon.calculate.shape.HandShapeAnalysisBuffer;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.Tile;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.prism.core.SuitSymmetry;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * 種を決めた半荘をエンジンで打ち、網に問う全ての判断（合法手が 2 つ以上）を順に渡す試験の道具。強制の判断も打つが、
 * {@link Visitor#visit} には渡さない（判断を待つ段としては {@link Visitor#awaiting} に渡す）。局の終わりは
 * {@link Visitor#settled} に渡す。
 *
 * <p>同じ種と打ち方なら同じ判断の列になる。渡した判断の手は、受け取り手が返した後に打ち方で選ぶ。
 */
public final class DecisionWalk {

  /** 打ち方。 */
  public enum Policy {
    /** 鳴き・槓・和了があれば選び（合法手の順で最後のもの）、なければ乱数。鳴きと槓が多く、出来事の多い局が出る。 */
    CALLS,
    /** 和了は取り、ほかは遷移の後の向聴が最小で受け入れの種類が最多の手。鳴きは向聴が進むときだけ、立直はできれば取る。 */
    GREEDY,
    /**
     * 和了・鳴き・槓・立直・九種九牌を取らず、么九牌があれば么九牌を（なければ手の牌を）乱数で捨てる。局は荒牌流局か途中流局で終わり、
     * 流し満貫が出る。
     */
    TERMINALS
  }

  /** 網に問う判断と、判断を待つ段を受け取る。 */
  @FunctionalInterface
  public interface Visitor {

    /**
     * 判断を受け取る。エンジンの状態を変えてはならない。
     *
     * @param engine 判断を待つエンジン（全ての状態は {@link GameEngine#getState()}）
     * @param point 判断（合法手が 2 つ以上）
     * @param round 局の通し番号（この walk の中で 0 始まり）
     */
    void visit(GameEngine engine, EngineDecisionPoint point, int round);

    /**
     * 判断を待つ段を、強制の判断だけの段も含めて、その段の判断を {@link #visit} に渡す前に受け取る。エンジンの状態を変えてはならない。
     * 既定は何もしない。
     *
     * @param engine 判断を待つエンジン
     * @param points その段の全ての判断（訪れない席の判断と強制の判断を含む）
     * @param round 局の通し番号
     */
    default void awaiting(GameEngine engine, List<EngineDecisionPoint> points, int round) {}

    /**
     * 局の終わり（精算の後、次局を始める前。エンジンの状態は局末のまま）を受け取る。半荘の最後の局も渡す。エンジンの状態を変えてはならない。
     * 既定は何もしない。
     *
     * @param engine 局を終えたエンジン
     * @param settlement その局の精算
     * @param round 局の通し番号
     */
    default void settled(GameEngine engine, RoundSettlement settlement, int round) {}
  }

  /** 萬子・筒子・索子を入れ替えた半荘と対にした判断を受け取る（{@link #runMirrored}）。 */
  @FunctionalInterface
  public interface MirrorVisitor {

    /**
     * 対の判断を受け取る。どちらのエンジンの状態も変えてはならない。
     *
     * @param engine 元の半荘のエンジン
     * @param point 元の判断（合法手が 2 つ以上）
     * @param mirror 色を入れ替えた半荘のエンジン
     * @param mirrorPoint 対の判断（同じ席。合法手は元の合法手を色の写しで写したもの）
     */
    void visit(
        GameEngine engine,
        EngineDecisionPoint point,
        GameEngine mirror,
        EngineDecisionPoint mirrorPoint);
  }

  private final Policy policy;
  private final EngineDecisionBuffer analysis = new EngineDecisionBuffer();
  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
  private final HandShapeAnalysisBuffer shape = new HandShapeAnalysisBuffer();
  private final HandShapeState afterShape = new HandShapeState();
  private final EngineSelectionBuffer selections = new EngineSelectionBuffer();
  private final EngineSelectionBuffer mirrorSelections = new EngineSelectionBuffer();

  /**
   * 道具を作る。
   *
   * @param policy 打ち方
   */
  public DecisionWalk(Policy policy) {
    this.policy = Objects.requireNonNull(policy, "policy");
  }

  /**
   * 半荘を打つ。半荘 {@code g} の牌山と打ち方の乱数の種は {@code seed + g} から決めるので、各半荘の判断の列は種の範囲によらない。
   *
   * @param seed 最初の半荘の種
   * @param games 半荘の数
   * @param visitor 判断を受け取る先
   */
  public void run(long seed, int games, Visitor visitor) {
    int round = 0;
    for (int game = 0; game < games; game++) {
      GameEngine engine = new GameEngine(seed + game);
      SplittableRandom random = new SplittableRandom(~(seed + game));
      GameStepResult step = engine.stepHanchan();
      while (!(step instanceof GameStepResult.HanchanEnded)) {
        if (!(step instanceof GameStepResult.AwaitingDecisions awaiting)) {
          if (step instanceof GameStepResult.RoundSettled settled) {
            visitor.settled(engine, settled.settlement(), round);
            round++;
          }
          step = engine.stepHanchan();
          continue;
        }
        selections.clear();
        List<EngineDecisionPoint> points = awaiting.decisions();
        visitor.awaiting(engine, points, round);
        for (int index = 0; index < points.size(); index++) {
          EngineDecisionPoint point = points.get(index);
          List<Action> legal = point.legalActions();
          if (legal.size() > 1) {
            visitor.visit(engine, point, round);
          }
          selections.add(point.id(), choose(engine.observation(point.player()), legal, random));
        }
        step = engine.commitDecisions(selections);
      }
      visitor.settled(engine, ((GameStepResult.HanchanEnded) step).settlement(), round);
      round++;
    }
  }

  /**
   * 半荘を、萬子・筒子・索子を入れ替えた牌山の半荘と対で打つ。手は元の半荘で打ち方が選び、対の半荘ではその手を色の写しで写した手を打つので、2 つの
   * 半荘は色の写しを除いて同じに進む。半荘 g の牌山と打ち方の乱数の種は {@link #run} と同じ。
   *
   * @param seed 最初の半荘の種
   * @param games 半荘の数
   * @param view 色の入れ替え（{@link SuitSymmetry} の番号）
   * @param visitor 対の判断を受け取る先
   * @throws IllegalStateException 2 つの半荘の進み方が食い違ったとき
   */
  public void runMirrored(long seed, int games, int view, MirrorVisitor visitor) {
    for (int game = 0; game < games; game++) {
      GameEngine engine = new GameEngine(seed + game);
      GameEngine mirror = new GameEngine(seed + game);
      mirror.getState().mapWallTiles(SuitSymmetry.physicalMap(view));
      SplittableRandom random = new SplittableRandom(~(seed + game));
      GameStepResult step = engine.stepHanchan();
      GameStepResult mirrorStep = mirror.stepHanchan();
      while (!(step instanceof GameStepResult.HanchanEnded)) {
        if (step.getClass() != mirrorStep.getClass()) {
          throw new IllegalStateException("mirrored hanchan diverged: " + step + " / " + mirrorStep);
        }
        if (!(step instanceof GameStepResult.AwaitingDecisions awaiting)) {
          step = engine.stepHanchan();
          mirrorStep = mirror.stepHanchan();
          continue;
        }
        List<EngineDecisionPoint> points = awaiting.decisions();
        List<EngineDecisionPoint> mirrorPoints =
            ((GameStepResult.AwaitingDecisions) mirrorStep).decisions();
        if (points.size() != mirrorPoints.size()) {
          throw new IllegalStateException("mirrored decisions diverged");
        }
        selections.clear();
        mirrorSelections.clear();
        for (int index = 0; index < points.size(); index++) {
          EngineDecisionPoint point = points.get(index);
          EngineDecisionPoint mirrorPoint = mirrorPoints.get(index);
          List<Action> legal = point.legalActions();
          List<Action> mirrorLegal = mirrorPoint.legalActions();
          if (point.player() != mirrorPoint.player() || legal.size() != mirrorLegal.size()) {
            throw new IllegalStateException("mirrored decision diverged at seat " + point.player());
          }
          if (legal.size() > 1) {
            visitor.visit(engine, point, mirror, mirrorPoint);
          }
          Action chosen = choose(engine.observation(point.player()), legal, random);
          int mapped = SuitSymmetry.find(view, mirrorLegal, chosen);
          if (mapped < 0) {
            throw new IllegalStateException("mirrored legal actions lack " + chosen);
          }
          selections.add(point.id(), chosen);
          mirrorSelections.add(mirrorPoint.id(), mirrorLegal.get(mapped));
        }
        step = engine.commitDecisions(selections);
        mirrorStep = mirror.commitDecisions(mirrorSelections);
      }
    }
  }

  private Action choose(PublicObservation o, List<Action> legal, SplittableRandom random) {
    if (legal.size() == 1) {
      return legal.get(0);
    }
    return switch (policy) {
      case CALLS -> calls(legal, random);
      case GREEDY -> greedy(o, legal, random);
      case TERMINALS -> terminals(legal, random);
    };
  }

  /** 見送りがあれば見送り、なければ么九牌の打牌（なければ全ての打牌）から乱数で 1 つ。 */
  private static Action terminals(List<Action> legal, SplittableRandom random) {
    int discards = 0;
    int terminals = 0;
    for (int index = 0; index < legal.size(); index++) {
      Action action = legal.get(index);
      if (action.type() == Action.Type.PASS) {
        return action;
      }
      if (action.type() == Action.Type.DAHAI) {
        discards++;
        terminals += Tile.isTerminalOrHonor(action.tileType()) ? 1 : 0;
      }
    }
    boolean onlyTerminals = terminals > 0;
    int pick = random.nextInt(onlyTerminals ? terminals : discards);
    for (int index = 0; index < legal.size(); index++) {
      Action action = legal.get(index);
      if (action.type() == Action.Type.DAHAI
          && (!onlyTerminals || Tile.isTerminalOrHonor(action.tileType()))
          && pick-- == 0) {
        return action;
      }
    }
    throw new IllegalStateException("no discard among " + legal);
  }

  private static Action calls(List<Action> legal, SplittableRandom random) {
    Action chosen = legal.get(random.nextInt(legal.size()));
    for (int index = 0; index < legal.size(); index++) {
      switch (legal.get(index).type()) {
        case CHI, PON, DAIMINKAN, ANKAN, KAKAN, RON_AGARI, TSUMO_AGARI -> chosen = legal.get(index);
        default -> {}
      }
    }
    return chosen;
  }

  private Action greedy(PublicObservation o, List<Action> legal, SplittableRandom random) {
    o.analyze(legal, analysis);
    int best = 0;
    long bestKey = Long.MIN_VALUE;
    for (int index = 0; index < legal.size(); index++) {
      Action.Type type = legal.get(index).type();
      if (type == Action.Type.RON_AGARI || type == Action.Type.TSUMO_AGARI) {
        return legal.get(index);
      }
      if (type == Action.Type.KYUSHU_KYUHAI) {
        continue;
      }
      long key = 8 * value(index, type) + random.nextInt(8);
      if (key > bestKey) {
        bestKey = key;
        best = index;
      }
    }
    return legal.get(best);
  }

  /** 手の値: 遷移の後の最もよい −1000·向聴 + 受け入れの種類。見送りに 500（鳴きは向聴が進むときだけ）、立直に 1 を足す。 */
  private long value(int action, Action.Type type) {
    long best = Long.MIN_VALUE / 16;
    for (int transition = 0; transition < analysis.transitionCount(action); transition++) {
      // 後の手の形を写してから解析する（解析器が見る手の型を、本番と同じ形と局面の手だけにして、測る側の JIT を乱さない）。
      afterShape.load(analysis.handAfterTransition(action, transition));
      shapes.analyzeInto(afterShape, shape);
      best =
          Math.max(
              best, -1000L * shape.minimumShanten() + Long.bitCount(shape.ukeireTileTypeMask()));
    }
    if (type == Action.Type.PASS) {
      best += 500;
    }
    if (type == Action.Type.RIICHI_DAHAI) {
      best += 1;
    }
    return best;
  }
}
