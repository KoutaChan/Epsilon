package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatTable;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.AllocationProbe;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 1 席から見た卓（四人打ちと三人打ち）のテスト。卓は他家のツモを伏せた記録を受け、自分が判断するたびに、同じ記録を再生したエンジンが
 * その席に見せるものと同じ卓と合法手を作る。局末には記録の結果を自分の卓で精算し、点の移動と次の卓がエンジンと一致する。もしもの局は、値の
 * 範囲だけを確かめて始める。
 */
public class SeatTableTest {

  /** 照合する自己対局の半荘の数（三人打ちで、抜いた北への応答が起きるのに足りる数）。 */
  private static final int GAMES = 100;

  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /** ルールごとに手で組み立てた半荘。自己対局の記録に加えて照合する。 */
  @DataProvider
  public static Object[][] records() {
    List<RecordedGame> fourPlayer =
        List.of(
            ReplayDriverTest.game(
                "pon-over-chi",
                ReplayDriverTest.ponOverChi().round(ReplayDriverTest.ponOverChiResult()),
                24_000,
                21_000,
                31_000,
                24_000),
            ReplayDriverTest.game(
                "kamicha-furiten",
                ReplayDriverTest.kamichaPon().round(ReplayDriverTest.furitenRon()),
                24_000,
                30_300,
                21_700,
                24_000),
            ReplayDriverTest.game(
                "toimen-furiten",
                ReplayDriverTest.toimenPon().round(ReplayDriverTest.furitenRon()),
                24_000,
                30_300,
                21_700,
                24_000));
    return new Object[][] {{Variant.TENHOU_4P, fourPlayer}, {Variant.TENHOU_3P, List.of()}};
  }

  /** 北を抜くルールでは、抜いた北への応答の判断も照合する。 */
  @Test(dataProvider = "records")
  public void eachSeatSeesWhatTheEngineShowsItAtEveryOwnDecision(Variant variant, List<RecordedGame> handMade) {
    List<RecordedGame> games = new ArrayList<>();
    for (int seed = 100; seed < 100 + GAMES; seed++) {
      games.add(RecordedGames.play(variant, seed, "self-" + seed));
    }
    games.addAll(handMade);
    int seats = variant.seats();
    ReplayDriver driver = new ReplayDriver(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    SeatTable[] tables = new SeatTable[seats];
    for (int seat = 0; seat < seats; seat++) {
      tables[seat] = new SeatTable(variant, seat);
    }
    RoundOutcome outcome = new RoundOutcome(variant);
    int compared = 0;
    int nukiResponses = 0;
    for (RecordedGame game : games) {
      for (RecordedRound round : game.rounds()) {
        List<List<String>> expected = new ArrayList<>();
        for (int seat = 0; seat < seats; seat++) {
          expected.add(new ArrayList<>());
        }
        Boundary boundary = driver.start(round, ws);
        while (boundary == Boundary.DECISIONS) {
          Decisions decisions = driver.engine().decisions();
          if (decisions.kind(0) == EngineDecisionKind.NUKI_RESPONSE) {
            nukiResponses += decisions.size();
          }
          for (int slot = 0; slot < decisions.size(); slot++) {
            int seat = decisions.seat(slot);
            expected.get(seat).add(snapshot(driver.engine().view(seat), decisions, slot));
          }
          boundary = driver.step(ws);
        }
        TableState next = driver.engine().roundEnd().next();
        round.result().outcome(round.riichiMask(), outcome);
        for (int seat = 0; seat < seats; seat++) {
          String where = game.meta().id() + " " + round.start() + " seat " + seat;
          List<String> seen = follow(tables[seat], round, ws);
          Assert.assertEquals(seen, expected.get(seat), where);
          RoundEnd end = tables[seat].settle(outcome);
          Assert.assertTrue(round.result().sameDeltas(end), where);
          Assert.assertEquals(end.next().toString(), next.toString(), where);
          compared += seen.size();
        }
      }
    }
    Assert.assertTrue(compared > 1_000, compared + " decisions");
    if (variant.has(RuleFlag.NUKI)) {
      Assert.assertTrue(nukiResponses > 0, "no response to a nuki was compared");
    }
  }

  /** 自分の打牌が自分の合法手に無ければ、その記録を受け付けない。 */
  @Test
  public void anOwnDiscardOutsideTheLegalActionsIsRejected() {
    RecordedRound round = ReplayDriverTest.ponOverChi().round(ReplayDriverTest.ponOverChiResult());
    Variant variant = round.start().variant();
    SeatTable table = new SeatTable(variant, 3);
    RuleWorkspace ws = new RuleWorkspace(variant);
    table.startRound(round.start(), hand(round, 3), round.indicator());
    round.events().get(0).applyTo(table, ws);
    RecordEvent missing = new RecordEvent.Discard(3, Tiles.physicalId(Tiles.S5, 1), false, false);
    Assert.assertThrows(IllegalArgumentException.class, () -> missing.applyTo(table, ws));
  }

  /**
   * もしもの局を局の始まりと同じ値で始めると、どの席から見ても、ふつうに始めた局と同じ卓と合法手になる。点と供託の合計が合わなくても
   * 始められる。席のビューの点・本場・供託は渡した値になり、局末の精算もその値から行う。範囲の外の値や、点の個数・配牌の枚数が違う
   * 呼び出しは拒み、拒んだときは卓を変えない。
   */
  @Test(dataProvider = "variants")
  public void aScenarioChecksOnlyTheRanges(Variant variant) {
    RecordedRound round = RecordedGames.play(variant, 3, "scenario").rounds().get(0);
    RoundStart start = round.start();
    RuleWorkspace ws = new RuleWorkspace(variant);
    for (int seat = 0; seat < variant.seats(); seat++) {
      List<String> expected = follow(new SeatTable(variant, seat), round, ws);
      SeatTable scenario = new SeatTable(variant, seat);
      scenario.startScenario(
          start.round(), start.honba(), start.kyotaku(), start.scores(), hand(round, seat), round.indicator());
      Assert.assertEquals(playOn(scenario, round, ws), expected, "seat " + seat);
    }

    int[] scores = new int[variant.seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = 50_000 - 10_000 * seat;
    }
    SeatTable table = new SeatTable(variant, 0);
    table.startScenario(start.round(), 255, 100, scores, hand(round, 0), round.indicator());
    scores[0] = 0;
    SeatView view = table.view();
    Assert.assertEquals(view.honba(), 255);
    Assert.assertEquals(view.kyotaku(), 100);
    for (int seat = 0; seat < scores.length; seat++) {
      Assert.assertEquals(view.score(seat), 50_000 - 10_000 * seat, "seat " + seat);
    }
    scores[0] = 50_000;
    Assert.assertThrows(
        IllegalArgumentException.class, () -> new RoundStart(variant, start.round(), 255, 100, scores));
    playOn(table, round, ws);
    RoundOutcome outcome = new RoundOutcome(variant);
    round.result().outcome(round.riichiMask(), outcome);
    RoundEnd end = table.settle(outcome);
    if (!end.next().gameOver()) {
      for (int seat = 0; seat < scores.length; seat++) {
        Assert.assertEquals(end.next().score(seat), 50_000 - 10_000 * seat + end.delta(seat), "seat " + seat);
      }
    }

    int[] hand = hand(round, 0);
    int indicator = round.indicator();
    int[] even = start.scores();
    SeatTable checked = new SeatTable(variant, 0);
    checked.startScenario(start.round(), 1, 2, even, hand, indicator);
    Assert.assertThrows(IllegalArgumentException.class, () -> checked.startScenario(-1, 0, 0, even, hand, indicator));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> checked.startScenario(variant.finalRound() + 1, 0, 0, even, hand, indicator));
    Assert.assertThrows(IllegalArgumentException.class, () -> checked.startScenario(0, -1, 0, even, hand, indicator));
    Assert.assertThrows(IllegalArgumentException.class, () -> checked.startScenario(0, 0, -1, even, hand, indicator));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> checked.startScenario(0, 0, 0, Arrays.copyOf(even, even.length + 1), hand, indicator));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> checked.startScenario(0, 0, 0, even, Arrays.copyOf(hand, Variant.HAND - 1), indicator));
    Assert.assertEquals(checked.view().honba(), 1, "a rejected scenario changes nothing");
    Assert.assertEquals(checked.view().kyotaku(), 2, "a rejected scenario changes nothing");
  }

  /** ウォームアップの後は、記録のイベントを受ける処理も局末の精算もメモリを確保しない。 */
  @Test(dataProvider = "variants")
  public void eventsAllocateNothingOnceWarm(Variant variant) {
    int seat = 2;
    List<RecordedRound> rounds = RecordedGames.play(variant, 7, "warm").rounds();
    int[][] hands = new int[rounds.size()][];
    List<List<RecordEvent>> shown = new ArrayList<>();
    RoundOutcome[] outcomes = new RoundOutcome[rounds.size()];
    for (int index = 0; index < rounds.size(); index++) {
      RecordedRound round = rounds.get(index);
      hands[index] = hand(round, seat);
      shown.add(shown(round, seat));
      outcomes[index] = new RoundOutcome(variant);
      round.result().outcome(round.riichiMask(), outcomes[index]);
    }
    SeatTable table = new SeatTable(variant, seat);
    RuleWorkspace ws = new RuleWorkspace(variant);
    AllocationProbe.assertNoAllocation(
        WARM_UP,
        MEASURED,
        events -> {
          long checksum = 0;
          int applied = 0;
          for (int index = 0; applied < events; index = (index + 1) % rounds.size()) {
            RecordedRound round = rounds.get(index);
            List<RecordEvent> recorded = shown.get(index);
            table.startRound(round.start(), hands[index], round.indicator());
            for (int event = 0; event < recorded.size(); event++) {
              recorded.get(event).applyTo(table, ws);
              checksum += table.decisions().size();
            }
            applied += recorded.size();
            checksum += table.settle(outcomes[index]).delta(seat);
          }
          return checksum;
        });
  }

  /**
   * 1 席から見た卓で局を始め、他家のツモを伏せた記録を受けさせて、判断ごとの卓と合法手を集める。判断は ID が新しくなったときだけ数える
   * （立直の成立では応答を締めないので、その後も同じ判断が残る）。
   */
  private static List<String> follow(SeatTable table, RecordedRound round, RuleWorkspace ws) {
    table.startRound(round.start(), hand(round, table.seat()), round.indicator());
    return playOn(table, round, ws);
  }

  /** すでに始めた局に記録のイベントを受けさせ、判断ごとの卓と合法手を集める。判断は ID が新しくなったときだけ数える。 */
  private static List<String> playOn(SeatTable table, RecordedRound round, RuleWorkspace ws) {
    int seat = table.seat();
    List<String> seen = new ArrayList<>();
    long lastId = 0;
    for (RecordEvent event : shown(round, seat)) {
      event.applyTo(table, ws);
      Decisions decisions = table.decisions();
      if (decisions.size() > 0 && decisions.id(0) != lastId) {
        lastId = decisions.id(0);
        seen.add(snapshot(table.view(), decisions, 0));
      }
    }
    return seen;
  }

  /** 見る席から見た記録。他家がツモった牌を伏せる。 */
  private static List<RecordEvent> shown(RecordedRound round, int seat) {
    List<RecordEvent> shown = new ArrayList<>();
    for (RecordEvent event : round.events()) {
      shown.add(
          event instanceof RecordEvent.Draw draw && draw.seat() != seat
              ? new RecordEvent.Draw(draw.seat(), TurnEvent.NONE)
              : event);
    }
    return List.copyOf(shown);
  }

  private static int[] hand(RecordedRound round, int seat) {
    int[] hand = new int[Variant.HAND];
    for (int index = 0; index < hand.length; index++) {
      hand[index] = round.deal(seat, index);
    }
    return hand;
  }

  /**
   * 見る席から読める卓の値の全てと、その席の合法手を並べた文字列。
   *
   * @param slot 見る席の判断が、判断する席のうち何番目か
   */
  private static String snapshot(SeatView view, Decisions decisions, int slot) {
    StringBuilder text = new StringBuilder();
    text.append(view.kind())
        .append(" round=").append(view.round())
        .append(" dealer=").append(view.dealer())
        .append(" current=").append(view.current())
        .append(" honba=").append(view.honba())
        .append(" kyotaku=").append(view.kyotaku())
        .append(" live=").append(view.liveLeft());
    for (int seat = 0; seat < view.variant().seats(); seat++) {
      text.append(" | seat ").append(seat)
          .append(' ').append(view.score(seat))
          .append(' ').append(view.riichi(seat))
          .append(view.ippatsu(seat) ? " ippatsu" : "")
          .append(" river=").append(Long.toHexString(view.riverKinds(seat)))
          .append(" safe=").append(Long.toHexString(view.riichiSafe(seat)))
          .append(" nuki=").append(view.nuki(seat))
          .append(" melds=");
      for (int index = 0; index < view.meldCount(seat); index++) {
        text.append(Meld.of(view.meld(seat, index))).append(';');
      }
    }
    EventLog events = view.events();
    text.append(" | events");
    for (int event = 0; event < events.size(); event++) {
      text.append(' ').append(events.kind(event)).append(events.seat(event))
          .append('/').append(events.tile(event))
          .append(events.red(event) ? "r" : "")
          .append(events.tsumogiri(event) ? "t" : "")
          .append(events.riichi(event) ? "R" : "")
          .append('/').append(events.calledBy(event))
          .append('/').append(events.meld(event));
    }
    text.append(" dahai=").append(events.dahai()).append(" interruptions=").append(events.interruptions());
    TurnEvent last = view.last();
    text.append(" | last ").append(last.kind()).append(last.seat())
        .append('/').append(last.tile())
        .append(last.red() ? "r" : "")
        .append(last.rinshan() ? " rinshan" : "")
        .append(last.doraPending() ? " pending" : "")
        .append(' ').append(last.kanKind());
    text.append(" | dora");
    for (int index = 0; index < view.indicatorCount(); index++) {
      text.append(' ').append(view.indicator(index)).append('@').append(view.indicatorEvent(index));
    }
    text.append(" | kinds");
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      text.append(' ').append(view.count(kind))
          .append('/').append(view.visible(kind))
          .append('/').append(view.doraWeight(kind));
    }
    text.append(" | held");
    for (int tile = 0; tile < Tiles.PHYSICAL_IDS; tile++) {
      if (view.holds(tile)) {
        text.append(' ').append(tile);
      }
    }
    text.append(" | furiten=").append(view.furiten()).append(" waits=").append(Long.toHexString(view.waits()));
    text.append(" | legal");
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      text.append(' ').append(Action.of(decisions.legal(slot, index)));
    }
    return text.toString();
  }
}
