package com.epsilon.engine;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayRecordReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.SplittableRandom;

/**
 * 試験が共有する決まった対局の組（エンジンの不変条件は {@link EngineInvariantsTest} が確かめる）。
 *
 * <p>組は 4 つで、この順に打つ（{@link Section}）。
 *
 * <ol>
 *   <li>一様: 種を決めた一様な手の選び方の半荘 {@value #UNIFORM_GAMES} 個。
 *   <li>偏り: 向聴を下げる打牌で聴牌を目指し、和了れれば和了り、加槓と立直は必ず、槓は高い確率でする打ち方の半荘 {@value #BIASED_GAMES} 個
 *       （立直・和了・ダブロンを一様より多く通す）。
 *   <li>場面: 稀な道を必ず通す作った局（{@link GoldenScenes}）。
 *   <li>牌譜: 保存した天鳳の牌譜を {@link ReplayEngine} で再生（Reviewer の道）。
 * </ol>
 *
 * <p>半荘の主経路では、境目のおよそ 8 回に 1 回で分岐し（{@link GameEngine#forkAtDecisionBoundary()}）、分岐は局末まで打ってから主経路に
 * 戻る（深さ優先）。分岐の半分は見えない牌を並べ替える（{@link GameState#reshuffleUnseen}）。半荘の一部は席別ツモ列と牌の写しを使う。場面は 4
 * 回に 1 回の境目で並べ替えずに分岐し、同じ台本で局末まで打つ（分岐が局面を正しく写すなら主経路と同じ結果になる）。
 *
 * <p>訪れる順・判断・乱数は種だけで決まる。訪問者は判断を読むだけで、エンジンの状態を変えてはならない。
 *
 * <p>半荘の組（一様・偏り）が稀な道を本当に通ったかは {@link Coverage} に数える（場面は場面ごとの期待で確かめる）。
 */
public final class GoldenGames {
  static final int UNIFORM_GAMES = 50;
  static final int BIASED_GAMES = 40;

  /** 保存した牌譜の置き場（test の resources）。 */
  public static final String RESOURCES = "/com/epsilon/engine/golden/";

  /**
   * 保存した天鳳の鳳凰卓の牌譜。double-ron はダブロンを含み、kakan は加槓が多い。打ち手の名・段位・レートは中立の値に替えてある（再生には
   * 使わない）。
   */
  private static final String[] RECORDS = {
    "double-ron-a.xml.gz", "double-ron-b.xml.gz", "kakan-a.xml.gz", "kakan-b.xml.gz"
  };

  /** 萬子→筒子→索子→萬子の写し（同じ順位・同じ枚目。赤は赤へ写る）。 */
  private static final int[] SUIT_ROTATION = suitRotation();

  private GoldenGames() {}

  /** 対局の組。 */
  public enum Section {
    UNIFORM,
    BIASED,
    SCENE,
    REPLAY
  }

  /** 半荘の組で数える道。 */
  public enum Path {
    /** 立直の宣言（実行した手）。 */
    RIICHI,
    /** ツモの和了。 */
    TSUMO,
    /** ロンの和了。 */
    RON,
    /** 2 人以上のロン。 */
    MULTIPLE_RON,
    /** 暗槓（実行した手）。 */
    ANKAN,
    /** 加槓（実行した手）。 */
    KAKAN,
    /** 大明槓（実行した手）。 */
    DAIMINKAN,
    /** 嶺上牌のツモの後の判断の段。 */
    RINSHAN_DRAW,
    /** 聴牌の席がある荒牌流局。 */
    TENPAI_DRAW,
    /** 途中流局。 */
    ABORTIVE_DRAW,
    /** 見えない牌を並べ替えない分岐。 */
    FORK,
    /** 見えない牌を並べ替えた分岐。 */
    RESHUFFLED_FORK,
    /** 席別ツモ列と牌の写しを両方使う半荘。 */
    STREAM_MAPPED_GAME
  }

  /** 半荘の組（主経路と分岐）で道を通った数。{@link #play} が数え、呼び手は読むだけ。 */
  public static final class Coverage {
    private final int[] counts = new int[Path.values().length];

    /** 道を通った数。 */
    public int count(Path path) {
      return counts[path.ordinal()];
    }

    private void add(Path path) {
      counts[path.ordinal()]++;
    }

    @Override
    public String toString() {
      StringBuilder text = new StringBuilder();
      for (Path path : Path.values()) {
        if (!text.isEmpty()) text.append(' ');
        text.append(path).append('=').append(count(path));
      }
      return text.toString();
    }
  }

  /** 判断の位置（組・対局・局・分岐）。訪問者の呼び出しの間だけ有効で、打つ側が書き換える。 */
  public static final class Position {
    private Section section;
    private int game;
    private int round;
    private int fork;

    public Section section() {
      return section;
    }

    /** 分岐の番号（主経路は 0）。 */
    public int fork() {
      return fork;
    }

    /** 位置を 1 語に詰める（組 4 ビット・対局 16・局 12・分岐 16・判断 ID の下位 16）。 */
    public long pack(long decisionId) {
      return (long) section.ordinal() << 60
          | (long) game << 44
          | (long) round << 32
          | (long) fork << 16
          | decisionId & 0xFFFF;
    }

    /** {@link #pack} した位置を読める形にする。 */
    public static String describe(long packed) {
      return Section.values()[(int) (packed >>> 60)]
          + " game "
          + (packed >>> 44 & 0xFFFF)
          + " round "
          + (packed >>> 32 & 0xFFF)
          + (((packed >>> 16) & 0xFFFF) == 0 ? " main" : " fork " + (packed >>> 16 & 0xFFFF))
          + " decision id (low 16 bits) "
          + (packed & 0xFFFF);
    }

    private void enter(Section section, int game) {
      this.section = section;
      this.game = game;
      round = 0;
      fork = 0;
    }
  }

  /** 判断と局の終わりを受け取る。 */
  public interface Visitor {

    /**
     * エンジンが待つ判断の段を、手を選ぶ前に受け取る（主経路と分岐の両方）。
     *
     * @param engine 判断を待つエンジン
     * @param points その段の全ての判断
     * @param at 位置
     */
    void decisions(GameEngine engine, List<EngineDecisionPoint> points, Position at);

    /**
     * 段の手を反映した後の、判断ごとの結果（直前に {@link #decisions} で同じエンジンから受けた段の結果。既定は何もしない）。
     *
     * @param engine 手を反映したエンジン
     * @param outcomes 判断ごとの結果
     */
    default void committed(GameEngine engine, List<EngineDecisionOutcome> outcomes) {}

    /** 局の精算（主経路と分岐）。既定は何もしない。 */
    default void settled(GameEngine engine, RoundSettlement settlement) {}

    /**
     * 牌譜の判断（再生の状態を読むだけ）。
     *
     * @param state 判断の直前の状態
     * @param seat 判断する席
     * @param legal 合法手
     * @param chosenSlot 牌譜の選んだ位置（未観測なら −1）
     * @param stateId 再生の状態の改訂番号
     * @param at 位置
     */
    void replayed(GameState state, int seat, List<Action> legal, int chosenSlot, int stateId, Position at);

    /** 牌譜の終局の点。既定は何もしない。 */
    default void replayEnded(int[] finalScores) {}
  }

  /**
   * 全ての組を決まった順に打つ。
   *
   * @return 半荘の組で道を通った数
   */
  public static Coverage play(Visitor visitor) {
    Position at = new Position();
    Coverage coverage = new Coverage();
    for (int game = 0; game < UNIFORM_GAMES; game++) {
      at.enter(Section.UNIFORM, game);
      playHanchan(Policy.UNIFORM, 1_000L + game, at, visitor, coverage);
    }
    for (int game = 0; game < BIASED_GAMES; game++) {
      at.enter(Section.BIASED, game);
      playHanchan(Policy.BIASED, 2_000L + game, at, visitor, coverage);
    }
    List<GoldenScenes.Scene> scenes = GoldenScenes.all();
    for (int index = 0; index < scenes.size(); index++) {
      at.enter(Section.SCENE, index);
      playScene(scenes.get(index), at, visitor);
    }
    for (int index = 0; index < RECORDS.length; index++) {
      at.enter(Section.REPLAY, index);
      int[] scores =
          ReplayEngine.replay(
              ReplayRecordReader.readRecord(recordBytes(RECORDS[index])),
              (point, state, seat, legal) ->
                  visitor.replayed(state, seat, legal, point.chosenSlot(), point.stateId(), at));
      visitor.replayEnded(scores);
    }
    return coverage;
  }

  /** 半荘の手の選び方。 */
  private enum Policy {
    UNIFORM,
    BIASED;

    /**
     * 判断の手を選ぶ。
     *
     * @param state 判断の局面（読むだけ）
     * @param point 判断
     * @param random 打ち方の乱数
     * @param shapes 打牌の後の向聴を求める解析器（呼び手のスレッドのもの）
     */
    Action choose(
        GameState state, EngineDecisionPoint point, SplittableRandom random, HandShapeAnalyzer shapes) {
      List<Action> legal = point.legalActions();
      return this == UNIFORM
          ? legal.get(random.nextInt(legal.size()))
          : biased(state, point.player(), legal, random, shapes);
    }

    /**
     * 和了 > 加槓 > 槓（8 割）> 立直 > 鳴き（3 割）> 九種九牌（5 割）> 打牌（打った後の向聴が最小の牌から一様）> 見送り > 一様。
     */
    private static Action biased(
        GameState state,
        int player,
        List<Action> legal,
        SplittableRandom random,
        HandShapeAnalyzer shapes) {
      Action pick = pick(legal, random, Action.Type.TSUMO_AGARI, Action.Type.RON_AGARI);
      if (pick == null) pick = pick(legal, random, Action.Type.KAKAN);
      if (pick == null && random.nextInt(10) < 8) {
        pick = pick(legal, random, Action.Type.ANKAN, Action.Type.DAIMINKAN);
      }
      if (pick == null) pick = pick(legal, random, Action.Type.RIICHI_DAHAI);
      if (pick == null && random.nextInt(10) < 3) {
        pick = pick(legal, random, Action.Type.PON, Action.Type.CHI);
      }
      if (pick == null && random.nextBoolean()) pick = pick(legal, random, Action.Type.KYUSHU_KYUHAI);
      if (pick == null) pick = efficientDiscard(state, player, legal, random, shapes);
      if (pick == null) pick = pick(legal, random, Action.Type.PASS);
      return pick != null ? pick : legal.get(random.nextInt(legal.size()));
    }

    /** 打った後の向聴が最小の打牌から一様に 1 つ（打牌がなければ null）。 */
    private static Action efficientDiscard(
        GameState state,
        int player,
        List<Action> legal,
        SplittableRandom random,
        HandShapeAnalyzer shapes) {
      int best = Integer.MAX_VALUE;
      int ties = 0;
      for (Action action : legal) {
        if (action.type() != Action.Type.DAHAI) continue;
        int shanten = shapes.calculateMinimumAfterRemoving(state.hand(player), action.tileType());
        if (shanten < best) {
          best = shanten;
          ties = 0;
        }
        if (shanten == best) ties++;
      }
      if (ties == 0) return null;
      int chosen = random.nextInt(ties);
      for (Action action : legal) {
        if (action.type() == Action.Type.DAHAI
            && shapes.calculateMinimumAfterRemoving(state.hand(player), action.tileType()) == best
            && chosen-- == 0) {
          return action;
        }
      }
      throw new AssertionError("unreachable");
    }

    /** 種類のどれかに当たる合法手から一様に 1 つ（なければ null）。 */
    private static Action pick(List<Action> legal, SplittableRandom random, Action.Type... types) {
      int count = 0;
      for (Action action : legal) if (matches(action, types)) count++;
      if (count == 0) return null;
      int chosen = random.nextInt(count);
      for (Action action : legal) {
        if (matches(action, types) && chosen-- == 0) return action;
      }
      throw new AssertionError("unreachable");
    }

    private static boolean matches(Action action, Action.Type[] types) {
      for (Action.Type type : types) if (action.type() == type) return true;
      return false;
    }
  }

  /** 1 つの半荘。席別ツモ列は対局 3 つに 1 つ、牌の写しは 5 つに 1 つ。 */
  private static void playHanchan(
      Policy policy, long seed, Position at, Visitor visitor, Coverage coverage) {
    GameEngine engine = new GameEngine(seed);
    boolean streams = at.game % 3 == 1;
    boolean mapped = at.game % 5 == 2;
    if (streams) engine.getState().enableSeatStreamDraws();
    if (mapped) engine.getState().mapWallTiles(SUIT_ROTATION);
    if (streams && mapped) coverage.add(Path.STREAM_MAPPED_GAME);
    SplittableRandom random = new SplittableRandom(GoldenHash.mix(seed));
    EngineSelectionBuffer selections = new EngineSelectionBuffer();
    HandShapeAnalyzer shapes = new HandShapeAnalyzer();
    int boundary = 0;
    int forks = 0;
    GameStepResult step = engine.stepHanchan();
    while (true) {
      switch (step) {
        case GameStepResult.HanchanEnded ended -> {
          settle(engine, ended.settlement(), visitor, coverage);
          return;
        }
        case GameStepResult.RoundSettled settled -> {
          settle(engine, settled.settlement(), visitor, coverage);
          at.round++;
          step = engine.stepHanchan();
        }
        case GameStepResult.AwaitingDecisions awaiting -> {
          visit(engine, awaiting.decisions(), at, visitor, coverage);
          long key = GoldenHash.mix(seed * 31 + boundary++);
          if ((key & 7) == 0) {
            at.fork = ++forks;
            playFork(engine, policy, key, (key >>> 3 & 1) == 0, at, visitor, coverage, shapes);
            at.fork = 0;
          }
          step =
              commit(
                  engine, awaiting.decisions(), policy, random, selections, shapes, visitor, coverage);
        }
        case GameStepResult.RoundEnded ended ->
            throw new IllegalStateException("hanchan stepping returned an unsettled round");
      }
    }
  }

  /** 境目で分岐し、局末まで打つ。{@code reshuffle} なら見えない牌を並べ替えてから。 */
  private static void playFork(
      GameEngine main,
      Policy policy,
      long key,
      boolean reshuffle,
      Position at,
      Visitor visitor,
      Coverage coverage,
      HandShapeAnalyzer shapes) {
    GameEngine fork = main.forkAtDecisionBoundary();
    if (reshuffle) fork.getState().reshuffleUnseen(GoldenHash.mix(key + 1));
    coverage.add(reshuffle ? Path.RESHUFFLED_FORK : Path.FORK);
    SplittableRandom random = new SplittableRandom(GoldenHash.mix(key + 2));
    EngineSelectionBuffer selections = new EngineSelectionBuffer();
    GameStepResult step = fork.stepHanchan();
    while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
      visit(fork, awaiting.decisions(), at, visitor, coverage);
      step = commit(fork, awaiting.decisions(), policy, random, selections, shapes, visitor, coverage);
    }
    settle(fork, settlement(step), visitor, coverage);
  }

  /** 半荘の判断の段を訪問者に渡し、嶺上牌のツモの後の段を数える。 */
  private static void visit(
      GameEngine engine,
      List<EngineDecisionPoint> points,
      Position at,
      Visitor visitor,
      Coverage coverage) {
    if (engine.getState().turnEvent() instanceof TurnEvent.Draw draw
        && draw.drawSource() == TurnEvent.DrawSource.RINSHAN) {
      coverage.add(Path.RINSHAN_DRAW);
    }
    visitor.decisions(engine, points, at);
  }

  private static GameStepResult commit(
      GameEngine engine,
      List<EngineDecisionPoint> points,
      Policy policy,
      SplittableRandom random,
      EngineSelectionBuffer selections,
      HandShapeAnalyzer shapes,
      Visitor visitor,
      Coverage coverage) {
    selections.clear();
    GameState state = engine.getState();
    for (EngineDecisionPoint point : points) {
      selections.add(point.id(), policy.choose(state, point, random, shapes));
    }
    EngineCommitResult result = engine.commitDecisionsWithOutcomes(selections);
    for (EngineDecisionOutcome outcome : result.decisionOutcomes()) {
      if (!outcome.selectedActionWasExecuted()) continue;
      switch (outcome.selectedAction().type()) {
        case RIICHI_DAHAI -> coverage.add(Path.RIICHI);
        case ANKAN -> coverage.add(Path.ANKAN);
        case KAKAN -> coverage.add(Path.KAKAN);
        case DAIMINKAN -> coverage.add(Path.DAIMINKAN);
        default -> {}
      }
    }
    visitor.committed(engine, result.decisionOutcomes());
    return result.step();
  }

  /** 半荘の局の精算を訪問者に渡し、局の結果の道を数える。 */
  private static void settle(
      GameEngine engine, RoundSettlement settlement, Visitor visitor, Coverage coverage) {
    switch (settlement.result()) {
      case RoundResult.TsumoAgari tsumo -> coverage.add(Path.TSUMO);
      case RoundResult.RonAgari ron -> {
        coverage.add(Path.RON);
        if (ron.claims().size() > 1) coverage.add(Path.MULTIPLE_RON);
      }
      case RoundResult.ExhaustiveDraw draw -> {
        if (draw.tenpaiMask() != 0) coverage.add(Path.TENPAI_DRAW);
      }
      case RoundResult.AbortiveDraw abort -> coverage.add(Path.ABORTIVE_DRAW);
    }
    visitor.settled(engine, settlement);
  }

  private static RoundSettlement settlement(GameStepResult step) {
    return switch (step) {
      case GameStepResult.RoundSettled settled -> settled.settlement();
      case GameStepResult.HanchanEnded ended -> ended.settlement();
      default -> throw new IllegalStateException("fork ended without a settlement: " + step);
    };
  }

  /** 場面の局（単独の局）。4 回に 1 回の境目で並べ替えずに分岐し、同じ台本で局末まで打つ。 */
  private static void playScene(GoldenScenes.Scene scene, Position at, Visitor visitor) {
    GameEngine engine = scene.start();
    EngineSelectionBuffer selections = new EngineSelectionBuffer();
    int boundary = 0;
    int forks = 0;
    GameStepResult step = engine.stepRound();
    while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
      visitor.decisions(engine, awaiting.decisions(), at);
      if (boundary++ % 4 == 3) {
        at.fork = ++forks;
        GameEngine fork = engine.forkAtDecisionBoundary();
        GameStepResult forked = fork.stepRound();
        while (forked instanceof GameStepResult.AwaitingDecisions pending) {
          visitor.decisions(fork, pending.decisions(), at);
          forked = commitScene(fork, pending.decisions(), scene, selections, visitor);
        }
        settleScene(fork, scene, forked, visitor);
        at.fork = 0;
      }
      step = commitScene(engine, awaiting.decisions(), scene, selections, visitor);
    }
    settleScene(engine, scene, step, visitor);
  }

  private static GameStepResult commitScene(
      GameEngine engine,
      List<EngineDecisionPoint> points,
      GoldenScenes.Scene scene,
      EngineSelectionBuffer selections,
      Visitor visitor) {
    selections.clear();
    for (EngineDecisionPoint point : points) selections.add(point.id(), scene.choose(engine, point));
    EngineCommitResult result = engine.commitDecisionsWithOutcomes(selections);
    visitor.committed(engine, result.decisionOutcomes());
    return result.step();
  }

  /** 場面の局を精算し、筋どおりの結果かを確かめる。 */
  private static void settleScene(
      GameEngine engine, GoldenScenes.Scene scene, GameStepResult step, Visitor visitor) {
    if (!(step instanceof GameStepResult.RoundEnded ended)) {
      throw new IllegalStateException(scene.name + ": round did not end: " + step);
    }
    if (!scene.expected(ended.result())) {
      throw new AssertionError(scene.name + ": unexpected result " + describe(ended.result()));
    }
    visitor.settled(engine, engine.settleRound(ended.result()));
  }

  private static String describe(RoundResult result) {
    return switch (result) {
      case RoundResult.TsumoAgari tsumo -> "tsumo by " + tsumo.winner() + " " + result.pointDelta();
      case RoundResult.RonAgari ron ->
          "ron from " + ron.loser() + " claims " + ron.claims().size() + " " + result.pointDelta();
      case RoundResult.ExhaustiveDraw draw ->
          "exhaustive draw tenpai " + draw.tenpaiMask() + " nagashi " + draw.nagashiWinnerMask();
      case RoundResult.AbortiveDraw abort -> "abortive draw " + abort.reason();
    };
  }

  /** 保存した牌譜の 1 つ（{@link #RESOURCES} の下の名前）のバイト。 */
  public static byte[] recordBytes(String name) {
    try (InputStream input = GoldenGames.class.getResourceAsStream(RESOURCES + name)) {
      if (input == null) throw new IllegalStateException("missing record " + name);
      return input.readAllBytes();
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  private static int[] suitRotation() {
    int[] map = new int[Tile.NUM_TILES];
    for (int id = 0; id < Tile.NUM_TILES; id++) {
      int type = Tile.typeOf(id);
      int rotated = type < Tile.TON ? (type + Tile.TILE_TYPES_PER_SUIT) % Tile.TON : type;
      map[id] = rotated * Tile.TILES_PER_TYPE + id % Tile.TILES_PER_TYPE;
    }
    return map;
  }
}
