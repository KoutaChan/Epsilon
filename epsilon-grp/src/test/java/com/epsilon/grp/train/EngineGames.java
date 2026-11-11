package com.epsilon.grp.train;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * epsilon-mahjong のエンジンで半荘を打つテスト用のヘルパー。打ち手は次の決まりと乱数で手を選ぶ。和了と北抜きは必ず選び、立直はできれば必ずかける。
 * 九種九牌は 1/4、鳴きは 1/3 の確率で選ぶ。打牌は向聴数を最も下げる牌から選ぶ。局ごとに、局の始まりの卓、エンジンが出した局の結果、
 * 精算の後の卓を返す。
 */
final class EngineGames {

  /**
   * 打った局 1 つ。
   *
   * @param start 局の始まりの卓
   * @param outcome エンジンが出した局の結果
   * @param next 精算の後の卓（半荘が終われば終局の卓）
   */
  record Hand(long gameId, TableState start, RoundOutcome outcome, TableState next) {}

  /**
   * 打った局と半荘の記録。どちらも打った順に並ぶ。
   */
  record Played(List<Hand> hands, List<GameRecord> games) {}

  private EngineGames() {}

  /**
   * @param firstId 最初の半荘 ID。以降は順に振る
   */
  static Played play(Variant variant, int games, long firstId, long seed) {
    Engine engine = new Engine(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    Player player = new Player(variant, engine, new Random(seed));
    List<Hand> hands = new ArrayList<>();
    List<GameRecord> records = new ArrayList<>(games);
    for (int game = 0; game < games; game++) {
      long gameId = firstId + game;
      engine.startGame(player.random.nextLong(), DrawOrder.WALL_ORDER);
      TableState start = new TableState(variant);
      List<GrpState> handEnds = new ArrayList<>();
      Boundary boundary = engine.advance(ws);
      while (true) {
        if (boundary == Boundary.DECISIONS) {
          boundary = engine.commit(player.choose(), ws);
          continue;
        }
        RoundEnd end = engine.roundEnd();
        RoundOutcome outcome = new RoundOutcome(variant);
        outcome.copyFrom(end.outcome());
        TableState next = new TableState(variant);
        next.copyFrom(end.next());
        hands.add(new Hand(gameId, start, outcome, next));
        if (boundary == Boundary.GAME_END) {
          int[] scores = new int[variant.seats()];
          for (int seat = 0; seat < scores.length; seat++) {
            scores[seat] = next.score(seat);
          }
          records.add(new GameRecord(variant, gameId, handEnds, scores));
          break;
        }
        handEnds.add(GrpState.of(next));
        start = next;
        boundary = engine.advance(ws);
      }
    }
    return new Played(hands, records);
  }

  /** 局を並んだ順に入れて、数のファイルの中身を作る。コホートは 4、探索の温度は 0 にする。 */
  static CountedHands counted(Variant variant, List<Hand> hands) {
    CountedHands.Builder builder = CountedHands.builder(variant, 4, 0, GrpFixtures.STAMP);
    for (Hand hand : hands) {
      builder.add(hand.gameId(), hand.start(), HandResult.raw(hand.outcome(), hand.start().dealer()));
    }
    return builder.build();
  }

  /** 乱数で手を選ぶ打ち手。 */
  private static final class Player {

    private final Engine engine;
    private final Random random;
    private final ShapeAnalyzer analyzer;
    private final HandShape shape = new HandShape();

    private Player(Variant variant, Engine engine, Random random) {
      this.engine = engine;
      this.random = random;
      analyzer = new ShapeAnalyzer(variant);
    }

    /** 判断する席ごとに手を選ぶ。選んだ手は、合法手の何番目かで返す。 */
    private int[] choose() {
      Decisions decisions = engine.decisions();
      int[] choices = new int[decisions.size()];
      for (int slot = 0; slot < choices.length; slot++) {
        choices[slot] = choose(decisions, slot);
      }
      return choices;
    }

    private int choose(Decisions decisions, int slot) {
      int win = -1;
      int kyushu = -1;
      int nuki = -1;
      int call = -1;
      int pass = -1;
      int riichi = -1;
      int riichis = 0;
      for (int index = 0; index < decisions.legalCount(slot); index++) {
        switch (Action.of(decisions.legal(slot, index)).type()) {
          case TSUMO_AGARI, RON_AGARI -> win = index;
          case KYUSHU_KYUHAI -> kyushu = index;
          case NUKI -> nuki = index;
          case CHI, PON, DAIMINKAN -> call = call < 0 ? index : call;
          case PASS -> pass = index;
          case RIICHI_DAHAI -> {
            riichi = riichi < 0 ? index : riichi;
            riichis++;
          }
          default -> {}
        }
      }
      if (win >= 0) {
        return win;
      }
      if (nuki >= 0) {
        return nuki;
      }
      if (kyushu >= 0 && random.nextInt(4) == 0) {
        return kyushu;
      }
      if (call >= 0 && random.nextInt(3) == 0) {
        return call;
      }
      if (pass >= 0) {
        return pass;
      }
      if (riichis > 0) {
        return riichi + random.nextInt(riichis);
      }
      return bestDiscard(decisions, slot);
    }

    /** 向聴数を最も下げる打牌を選ぶ。同じ向聴数の打牌が複数あれば一様に選び、打牌が無ければ合法手から一様に選ぶ。 */
    private int bestDiscard(Decisions decisions, int slot) {
      int seat = decisions.seat(slot);
      SeatView view = engine.view(seat);
      shape.clear();
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        for (int copy = 0; copy < view.count(kind); copy++) {
          shape.add(kind);
        }
      }
      for (int index = 0; index < view.meldCount(seat); index++) {
        shape.addMeld();
      }
      int best = Integer.MAX_VALUE;
      int chosen = -1;
      int ties = 0;
      for (int index = 0; index < decisions.legalCount(slot); index++) {
        Action action = Action.of(decisions.legal(slot, index));
        if (action.type() != Action.Type.DAHAI) {
          continue;
        }
        shape.remove(action.tile());
        int shanten = analyzer.shanten(shape);
        shape.add(action.tile());
        if (shanten < best) {
          best = shanten;
          chosen = index;
          ties = 1;
        } else if (shanten == best && random.nextInt(++ties) == 0) {
          chosen = index;
        }
      }
      return chosen < 0 ? random.nextInt(decisions.legalCount(slot)) : chosen;
    }
  }
}
