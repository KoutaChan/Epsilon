package com.epsilon.prism;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RandomGames;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.core.SuitSymmetry;
import java.util.Arrays;
import java.util.Random;

/**
 * 新しいエンジンのテストで卓を打ち進めるヘルパー。乱数の種で決まる半荘を、指定した打ち方で打つ。判断の時点では判断する席を全て渡し、
 * 局末も渡す。合法手が 1 つしかない判断も渡す。
 *
 * <p>色を入れ替えた対の局では、テストで作った牌山と、その色を入れ替えた牌山を {@code startRecordedRound} に渡して並べて打つ。入れ替えた
 * 側では、元の側が選んだ手の色を入れ替えた手を打つ。
 */
public final class TableWalk {

  /** 打ち方。 */
  public enum Policy {
    /** 鳴きと槓を選ぶ（公開のイベントが多い局と、槍槓を作る）。 */
    CALLS,
    /** 向聴数が下がる打牌と立直を選び、鳴かない。 */
    GREEDY,
    /** 中張牌から捨て、么九牌と字牌を残す（国士・混全帯・九種九牌の手を作る）。 */
    TERMINALS
  }

  /** 判断の時点と局末を受け取る。エンジンの状態を変えてはならない（写しは作ってよい）。 */
  public interface Visitor {
    /** 判断の時点で、判断する席を 1 つずつ受け取る。 */
    void decision(Engine engine, int slot);

    /** 局末（次の局を配る前）に呼ばれる。 */
    default void roundEnd(Engine engine) {}
  }

  /** 色を入れ替えた対の局で、同じ席の判断を受け取る。 */
  @FunctionalInterface
  public interface MirrorVisitor {
    void decision(Engine engine, int slot, Engine mirror, int mirrorSlot);
  }

  /** この回数に 1 回は、合法手から一様に選ぶ（同じ形ばかりにならないように）。 */
  private static final int EXPLORE = 12;

  private static final int WIN = 1_000;
  private static final int KAN = 900;
  private static final int CALL = 800;
  private static final int RIICHI = 700;
  private static final int NINE_TERMINALS = 600;
  private static final int PASS = 500;
  private static final int DISCARD = 400;
  private static final int SIMPLE = 20;
  private static final int PER_SHANTEN = 10;

  private final Variant variant;
  private final Policy policy;
  private final ShapeAnalyzer shapes;
  private final HandShape hand = new HandShape();
  private final HandShape after = new HandShape();
  private long decisions;
  private long rounds;

  public TableWalk(Variant variant, Policy policy) {
    this.variant = variant;
    this.policy = policy;
    shapes = new ShapeAnalyzer(variant);
  }

  /** 渡した判断の数（作ってからの合計）。 */
  public long decisions() {
    return decisions;
  }

  /** 打ち終えた局の数（作ってからの合計）。 */
  public long rounds() {
    return rounds;
  }

  /**
   * 乱数の種から半荘を続けて打つ。
   *
   * @param seed 半荘と手の選択に使う乱数の種
   * @param games 打つ半荘の数
   * @param visitor 判断と局末を受け取る先
   */
  public void run(long seed, int games, Visitor visitor) {
    RandomGames play = RandomGames.start(variant, seed, this::choose);
    int finished = 0;
    while (finished < games) {
      switch (play.boundary()) {
        case DECISIONS -> {
          Decisions boundary = play.engine.decisions();
          for (int slot = 0; slot < boundary.size(); slot++) {
            visitor.decision(play.engine, slot);
            decisions++;
          }
        }
        case ROUND_END -> {
          visitor.roundEnd(play.engine);
          rounds++;
        }
        case GAME_END -> {}
      }
      if (play.step() == Boundary.GAME_END) {
        finished++;
      }
    }
  }

  /**
   * 乱数の種から作った卓と牌山の局と、その色を入れ替えた局を並べて打つ。入れ替えた側で判断する席が元の側と違うとき、または入れ替えた側の
   * 合法手が元の側の合法手の色を入れ替えたものでないときは、AssertionError を投げる。
   *
   * @param seed 卓・牌山・手の選択に使う乱数の種
   * @param count 打つ局の数
   * @param view 色の入れ替え方（{@link SuitSymmetry} の番号）
   * @param visitor 対の判断を受け取る先
   */
  public void runMirrored(long seed, int count, int view, MirrorVisitor visitor) {
    Random random = new Random(seed);
    RuleWorkspace ws = new RuleWorkspace(variant);
    RuleWorkspace mirrorWs = new RuleWorkspace(variant);
    int[] choices = new int[variant.seats() - 1];
    int[] mirrorChoices = new int[variant.seats() - 1];
    for (int round = 0; round < count; round++) {
      RoundStart start = randomStart(random);
      int[] wall = randomWall(random);
      Engine engine = new Engine(variant);
      engine.startRecordedRound(start, wall);
      Engine mirror = new Engine(variant);
      mirror.startRecordedRound(start, SuitSymmetry.wall(view, wall));
      Boundary boundary = engine.advance(ws);
      Boundary mirrored = mirror.advance(mirrorWs);
      while (boundary == Boundary.DECISIONS) {
        if (mirrored != Boundary.DECISIONS) {
          throw new AssertionError("The mirrored round stopped at " + mirrored);
        }
        Decisions table = engine.decisions();
        Decisions shadow = mirror.decisions();
        if (table.size() != shadow.size()) {
          throw new AssertionError("The mirrored round has " + shadow.size() + " slots, not " + table.size());
        }
        for (int slot = 0; slot < table.size(); slot++) {
          int mirrorSlot = shadow.slotOf(table.seat(slot));
          if (mirrorSlot < 0 || table.legalCount(slot) != shadow.legalCount(mirrorSlot)) {
            throw new AssertionError("The mirrored round has other decisions of seat " + table.seat(slot));
          }
          visitor.decision(engine, slot, mirror, mirrorSlot);
          decisions++;
          choices[slot] = choose(engine, slot, random);
          Action chosen = Action.of(table.legal(slot, choices[slot]));
          mirrorChoices[mirrorSlot] = SuitSymmetry.find(view, shadow, mirrorSlot, chosen);
          if (mirrorChoices[mirrorSlot] < 0) {
            throw new AssertionError("The mirrored round has no image of " + chosen);
          }
        }
        boundary = engine.commit(choices, ws);
        mirrored = mirror.commit(mirrorChoices, mirrorWs);
      }
      if (mirrored != boundary) {
        throw new AssertionError("The mirrored round ended at " + mirrored + ", not " + boundary);
      }
      rounds++;
    }
  }

  /** 打ち方に従って、判断する席の手を選ぶ。値が同じ手からは乱数で選ぶ。 */
  private int choose(Engine engine, int slot, Random random) {
    Decisions boundary = engine.decisions();
    int legal = boundary.legalCount(slot);
    if (legal == 1) {
      return 0;
    }
    if (random.nextInt(EXPLORE) == 0) {
      return random.nextInt(legal);
    }
    engine.full().copyHand(boundary.seat(slot), hand);
    int best = 0;
    int bestValue = Integer.MIN_VALUE;
    for (int index = 0; index < legal; index++) {
      int value = value(Action.of(boundary.legal(slot, index))) * PER_SHANTEN + random.nextInt(PER_SHANTEN);
      if (value > bestValue) {
        best = index;
        bestValue = value;
      }
    }
    return best;
  }

  private int value(Action action) {
    return switch (action.type()) {
      case TSUMO_AGARI, RON_AGARI -> WIN;
      case ANKAN, KAKAN, DAIMINKAN -> policy == Policy.CALLS ? KAN : 0;
      case CHI, PON -> policy == Policy.CALLS ? CALL : 0;
      case PASS -> policy == Policy.CALLS ? 0 : PASS;
      case KYUSHU_KYUHAI -> policy == Policy.TERMINALS ? 0 : NINE_TERMINALS;
      case RIICHI_DAHAI -> policy == Policy.GREEDY ? RIICHI - shantenAfter(action.tile()) : 0;
      case DAHAI -> {
        boolean simple = (Tiles.bit(action.tile()) & Tiles.TERMINALS_AND_HONORS) == 0L;
        yield DISCARD - shantenAfter(action.tile()) + (policy == Policy.TERMINALS && simple ? SIMPLE : 0);
      }
      case NUKI -> policy == Policy.TERMINALS ? 0 : CALL;
    };
  }

  private int shantenAfter(int kind) {
    after.copyFrom(hand);
    after.remove(kind);
    return shapes.shanten(after);
  }

  private RoundStart randomStart(Random random) {
    int seats = variant.seats();
    int kyotaku = random.nextInt(3);
    int[] scores = new int[seats];
    Arrays.fill(scores, variant.startPoints());
    for (int deposit = 0; deposit < kyotaku; deposit++) {
      scores[random.nextInt(seats)] -= Variant.DEPOSIT;
    }
    for (int transfer = 0; transfer < seats; transfer++) {
      int from = random.nextInt(seats);
      int amount = 100 * random.nextInt(150);
      if (scores[from] >= amount) {
        scores[from] -= amount;
        scores[random.nextInt(seats)] += amount;
      }
    }
    return new RoundStart(variant, random.nextInt(variant.finalRound() + 1), random.nextInt(4), kyotaku, scores);
  }

  private int[] randomWall(Random random) {
    int[] wall = new int[variant.tiles().size()];
    for (int position = 0; position < wall.length; position++) {
      wall[position] = variant.tiles().physicalId(position);
    }
    for (int index = wall.length - 1; index > 0; index--) {
      int other = random.nextInt(index + 1);
      int tile = wall[index];
      wall[index] = wall[other];
      wall[other] = tile;
    }
    return wall;
  }
}
