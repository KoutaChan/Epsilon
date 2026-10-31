package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.TurnEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 手番の数え上げの不変条件: 各席の打牌 = 牌山のツモ + チー・ポン・大明槓（局を終えた手番だけは打牌しないことがある）、牌山のツモの合計 ≤ 70。
 * 河の長さに上限を置かず、鳴きの多い局でも手順が崩れないことを確かめる。
 *
 * <p>random は鳴きを高い率で受ける乱数の打ち手。target は席0の手番が最も増える鳴き方だけをする: 席0の捨て牌を他の3席がポン・チーし（上家の
 * ポンなら他の2席の手番が飛ぶ）、席0は下家の捨て牌をポンする。局数はシステムプロパティ turns.games で増やせ、turns.verbose で分布を出す。
 */
public class TurnAccountingTest {
  private static final Logger log = LoggerFactory.getLogger(TurnAccountingTest.class);

  @Test
  public void randomCallHeavyPlay() {
    Assert.assertEquals(run(false, Integer.getInteger("turns.games", 200), 0.95), "violations=0");
  }

  @Test
  public void playMaximizingOneSeatsTurns() {
    Assert.assertEquals(run(true, Integer.getInteger("turns.games", 200), 0), "violations=0");
  }

  private static String run(boolean target, int games, double callRate) {
    long seed = Long.getLong("turns.seed", 1);
    int hands = 0, violations = 0;
    int[] histogram = new int[64];
    var random = new SplittableRandom(seed);
    String example = null, longestExample = null;
    int longestSeen = 0;
    for (int game = 0; game < games; game++) {
      var engine = new GameEngine(seed * 1_000_003 + game);
      GameState state = engine.getState();
      int[] liveDraws = new int[4];
      GameStepResult step = engine.stepHanchan();
      while (!(step instanceof GameStepResult.HanchanEnded)) {
        if (step instanceof GameStepResult.AwaitingDecisions awaiting) {
          List<EngineDecisionPoint> points = awaiting.decisions();
          EngineDecisionPoint first = points.getFirst();
          if (first.kind() == EngineDecisionKind.TURN
              && state.getTurnEvent() instanceof TurnEvent.Draw draw
              && draw.player() == first.player()
              && !draw.isRinshanDraw()) liveDraws[draw.player()]++;
          int discarder =
              state.getTurnEvent() instanceof TurnEvent.Discard discard ? discard.player() : -1;
          List<EngineDecisionSelection> selections = new ArrayList<>();
          for (EngineDecisionPoint point : points) {
            Action action =
                target
                    ? targeted(point, discarder, random)
                    : choose(point.legalActions(), random, callRate);
            selections.add(new EngineDecisionSelection(point.id(), action));
          }
          step = engine.commitDecisions(selections);
        } else if (step instanceof GameStepResult.RoundSettled) {
          hands++;
          int longest = 0, longestSeat = 0;
          int totalLive = Arrays.stream(liveDraws).sum();
          for (int seat = 0; seat < 4; seat++) {
            int discards = state.river(seat).size();
            if (discards > longest) {
              longest = discards;
              longestSeat = seat;
            }
            int turns = liveDraws[seat] + turnCalls(state, seat);
            if (discards != turns && discards != turns - 1) {
              violations++;
              if (example == null)
                example =
                    "seat " + seat + " discards " + discards + " liveDraws " + liveDraws[seat]
                        + " calls " + turnCalls(state, seat);
            }
          }
          if (totalLive > 70) violations++;
          histogram[longest]++;
          if (longest > longestSeen) {
            longestSeen = longest;
            longestExample =
                "seat " + longestSeat + " discards " + longest + " (liveDraws "
                    + liveDraws[longestSeat] + ", calls " + turnCalls(state, longestSeat)
                    + "), live draws " + Arrays.toString(liveDraws) + " = " + totalLive
                    + ", river " + state.river(longestSeat);
          }
          Arrays.fill(liveDraws, 0);
          step = engine.stepHanchan();
        } else {
          throw new IllegalStateException(String.valueOf(step));
        }
      }
    }
    var out = new StringBuilder();
    out.append(
        String.format(
            "mode=%s hands=%d violations=%d%n", target ? "target" : "random", hands, violations));
    for (int length = 18; length < histogram.length; length++)
      if (histogram[length] > 0)
        out.append(String.format("  longest river %d: %d hands%n", length, histogram[length]));
    if (example != null) out.append("  violation e.g. ").append(example).append('\n');
    out.append("  longest: ").append(longestExample).append('\n');
    if (Boolean.getBoolean("turns.verbose")) log.info("{}", out.toString().stripTrailing());
    return violations == 0 ? "violations=0" : out.toString();
  }

  /** 手番を始めた鳴き（チー・ポン・大明槓）の数。加槓はポンの面子のまま数える。 */
  private static int turnCalls(GameState state, int seat) {
    int calls = 0;
    var hand = state.hand(seat);
    for (int index = 0; index < hand.meldCount(); index++)
      if (hand.meld(index).type() != Meld.Type.ANKAN) calls++;
    return calls;
  }

  /** 和了・立直・九種九牌はしない（局を長くする）。鳴きは callRate で受ける。打牌は一様。 */
  private static Action choose(List<Action> legal, SplittableRandom random, double callRate) {
    List<Action> calls = new ArrayList<>(), others = new ArrayList<>();
    Action pass = null;
    for (Action action : legal) {
      switch (action.type()) {
        case TSUMO_AGARI, RON_AGARI, RIICHI_DAHAI, KYUSHU_KYUHAI -> {}
        case CHI, PON, DAIMINKAN, ANKAN, KAKAN -> calls.add(action);
        case PASS -> pass = action;
        default -> others.add(action);
      }
    }
    if (!calls.isEmpty() && random.nextDouble() < callRate)
      return calls.get(random.nextInt(calls.size()));
    if (pass != null) return pass;
    if (!others.isEmpty()) return others.get(random.nextInt(others.size()));
    return legal.get(random.nextInt(legal.size()));
  }

  /** 席0の手番を増やす鳴きだけを受ける。槓はしない（牌山が短くなる）。 */
  private static Action targeted(EngineDecisionPoint point, int discarder, SplittableRandom random) {
    List<Action> legal = point.legalActions();
    int seat = point.player();
    boolean helps =
        discarder == 0 && seat != 0 // 席0の捨て牌を鳴く（上家=席3のポンなら席1・2が飛ぶ）
            || discarder == 1 && seat == 0; // 席0が下家の捨て牌をポンする（席2・3が飛ぶ）
    Action pass = null;
    List<Action> calls = new ArrayList<>(), discards = new ArrayList<>();
    for (Action action : legal) {
      switch (action.type()) {
        case PON -> calls.add(action);
        case CHI -> {
          if (seat != 0) calls.add(action);
        }
        case PASS -> pass = action;
        case DAHAI -> discards.add(action);
        default -> {}
      }
    }
    if (helps && !calls.isEmpty()) return calls.get(random.nextInt(calls.size()));
    if (pass != null) return pass;
    if (!discards.isEmpty()) return discards.get(random.nextInt(discards.size()));
    return legal.get(random.nextInt(legal.size()));
  }
}
