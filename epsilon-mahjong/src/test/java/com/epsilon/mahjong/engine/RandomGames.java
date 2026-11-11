package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;
import java.util.Random;

/**
 * 半荘を続けて打ち進める、テスト用のヘルパー（ほかのモジュールのテストも使う）。判断の時点では、判断する席ごとに選び方で手を
 * 選ぶ。局末では次の局を配り、終局では次の半荘を始める。手を選ぶ乱数と半荘の乱数の種は、作るときに渡す種で決まる。選び方が
 * メモリを確保しなければ、ウォームアップの後はメモリを確保しない。
 */
public final class RandomGames {

  /** 判断の時点で、判断する席の手を選ぶ。 */
  @FunctionalInterface
  public interface Chooser {

    /**
     * 判断する席の手を選ぶ。エンジンの状態を変えてはならない。
     *
     * @param engine 判断の時点で止まっているエンジン
     * @param slot 判断する席のうち何番目か
     * @param random 打ち進めるときに使う乱数
     * @return 選ぶ合法手の何番目か
     */
    int choose(Engine engine, int slot, Random random);
  }

  /** 合法手から一様に選ぶ。 */
  public static final Chooser UNIFORM =
      (engine, slot, random) -> random.nextInt(engine.decisions().legalCount(slot));

  public final Engine engine;
  public final RuleWorkspace ws;
  private final Chooser chooser;
  private final Random random;
  private final int[] choices;
  private Boundary boundary;

  /**
   * 止まっているエンジンを、合法手から一様に選んで打ち進める。
   *
   * @param engine 打ち進めるエンジン
   * @param boundary エンジンが止まっている時点
   * @param seed 手の選択と新しい半荘の種を決める乱数の種
   */
  public RandomGames(Engine engine, Boundary boundary, long seed) {
    this(engine, boundary, seed, UNIFORM);
  }

  /**
   * 止まっているエンジンを、渡した選び方で打ち進める。
   *
   * @param engine 打ち進めるエンジン
   * @param boundary エンジンが止まっている時点
   * @param seed 手の選択と新しい半荘の種を決める乱数の種
   * @param chooser 判断する席ごとの選び方
   */
  public RandomGames(Engine engine, Boundary boundary, long seed, Chooser chooser) {
    Variant variant = engine.full().variant();
    this.engine = engine;
    this.boundary = Objects.requireNonNull(boundary, "boundary");
    this.chooser = Objects.requireNonNull(chooser, "chooser");
    ws = new RuleWorkspace(variant);
    random = new Random(seed);
    choices = new int[variant.seats() - 1];
  }

  /** 新しいエンジンで半荘を始め、最初の判断の時点まで進める。手は合法手から一様に選ぶ。 */
  public static RandomGames start(Variant variant, long seed) {
    return start(variant, seed, UNIFORM);
  }

  /**
   * 新しいエンジンで半荘を始め、最初の判断の時点まで進める。
   *
   * @param variant 四人打ちか三人打ちか
   * @param seed 手の選択と半荘の種を決める乱数の種
   * @param chooser 判断する席ごとの選び方
   */
  public static RandomGames start(Variant variant, long seed, Chooser chooser) {
    RandomGames games = new RandomGames(new Engine(variant), Boundary.GAME_END, seed, chooser);
    games.step();
    return games;
  }

  /** いま止まっている時点（判断の時点・局末・終局）。 */
  public Boundary boundary() {
    return boundary;
  }

  /** 1 ステップ進めて、止まった時点を返す。判断の時点なら選んだ手で進め、局末なら次の局を配り、終局なら次の半荘を始める。 */
  public Boundary step() {
    switch (boundary) {
      case DECISIONS -> {
        Decisions decisions = engine.decisions();
        for (int slot = 0; slot < decisions.size(); slot++) {
          choices[slot] = chooser.choose(engine, slot, random);
        }
        boundary = engine.commit(choices, ws);
      }
      case ROUND_END -> boundary = engine.advance(ws);
      case GAME_END -> {
        engine.startGame(random.nextLong(), random.nextBoolean() ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
        boundary = engine.advance(ws);
      }
    }
    return boundary;
  }
}
