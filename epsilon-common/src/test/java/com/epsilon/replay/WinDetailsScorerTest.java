package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.GoldenGames;
import com.epsilon.replay.ReplayEvent.Hora;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 牌譜に内訳の無い和了は、実対局と同じ規則で採点し、牌譜の点の移動を再現した時だけ内訳を補う。 */
public class WinDetailsScorerTest {

  @DataProvider
  public Object[][] goldenRecords() {
    List<Object[]> cases = new ArrayList<>();
    for (String name : GoldenGames.records()) {
      cases.add(new Object[] {name, true});
      cases.add(new Object[] {name, false});
    }
    return cases.toArray(Object[][]::new);
  }

  /** 天鳳の牌譜から内訳を消して再生すると、全ての和了（ダブロンを含む）に天鳳の内訳と同じ物を補う。牌山が無い組み立て直しの道も同じ。 */
  @Test(dataProvider = "goldenRecords")
  public void computedDetailsMatchTenhouForEveryWin(String name, boolean walls) {
    ReplayRecord record = read(name);
    Map<Integer, WinDetails> recorded = recordedWins(record);
    Map<Integer, WinDetails> settled = settledWins(withoutScoring(record, walls));
    Assert.assertFalse(recorded.isEmpty(), name);
    Assert.assertEquals(settled.keySet(), recorded.keySet(), name);
    for (var win : recorded.entrySet())
      assertSameScoring(settled.get(win.getKey()), win.getValue(), name + " event " + win.getKey());
  }

  @Test
  public void winsWhosePointsTheRulesCannotReproduceKeepWhatTheRecordShows() {
    ReplayRecord record = withoutScoring(read(GoldenGames.records().getFirst()), true);
    List<ReplayEvent> events = new ArrayList<>(record.events());
    int index = firstWin(events);
    Hora win = (Hora) events.get(index);
    int[] deltas = win.deltas().clone();
    int payer = win.actor() != win.target() ? win.target() : (win.actor() + 1) % 4;
    deltas[win.actor()] += 100;
    deltas[payer] -= 100;
    events.set(
        index, new Hora(win.actor(), win.target(), win.winTileType(), deltas, win.details()));
    var changed = new ReplayRecord(events, record.metadata(), record.walls(), record.completion());
    Assert.assertEquals(settledWins(changed).get(index), win.details());
  }

  @Test
  public void winsWithRecordedScoringAreNotRecomputed() {
    ReplayRecord record = read(GoldenGames.records().getFirst());
    Assert.assertEquals(settledWins(record), recordedWins(record));
  }

  private static void assertSameScoring(WinDetails actual, WinDetails expected, String where) {
    Assert.assertNotNull(actual, where);
    Assert.assertTrue(actual.scored(), where);
    Assert.assertEquals(actual.han(), expected.han(), where);
    if (expected.yakuman() == 0) Assert.assertEquals(actual.fu(), expected.fu(), where);
    Assert.assertEquals(actual.points(), expected.points(), where);
    Assert.assertEquals(actual.yakuman(), expected.yakuman(), where);
    Assert.assertEquals(byCode(actual.yaku()), byCode(expected.yaku()), where);
    Assert.assertEquals(actual.uraIndicators(), expected.uraIndicators(), where);
  }

  private static List<WinDetails.Yaku> byCode(List<WinDetails.Yaku> yaku) {
    return yaku.stream().sorted(Comparator.comparing(WinDetails.Yaku::code)).toList();
  }

  private static ReplayRecord read(String name) {
    return ReplayRecordReader.readRecord(GoldenGames.recordBytes(name));
  }

  /** 和了の内訳を、牌譜に無かった形（裏ドラ表示牌だけ）にする。walls が偽なら牌山も消す（mjai と同じ組み立て直しの道）。 */
  private static ReplayRecord withoutScoring(ReplayRecord record, boolean walls) {
    List<ReplayEvent> events = new ArrayList<>();
    for (ReplayEvent event : record.events())
      events.add(event instanceof Hora win ? uraOnly(win) : event);
    return new ReplayRecord(
        events, record.metadata(), walls ? record.walls() : null, record.completion());
  }

  private static Hora uraOnly(Hora win) {
    WinDetails details = win.details();
    WinDetails ura =
        details == null || details.uraIndicators() == null
            ? null
            : new WinDetails(null, null, null, null, null, details.uraIndicators());
    return new Hora(win.actor(), win.target(), win.winTileType(), win.deltas(), ura);
  }

  private static Map<Integer, WinDetails> recordedWins(ReplayRecord record) {
    Map<Integer, WinDetails> wins = new TreeMap<>();
    List<ReplayEvent> events = record.events();
    for (int index = 0; index < events.size(); index++)
      if (events.get(index) instanceof Hora win) wins.put(index, win.details());
    return wins;
  }

  /** 再生が通知した和了の内訳（出来事の番号ごと）。 */
  private static Map<Integer, WinDetails> settledWins(ReplayRecord record) {
    Map<Integer, WinDetails> wins = new TreeMap<>();
    ReplayEngine.replay(
        record,
        new ReplayObserver() {
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {}

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (event instanceof Hora win) wins.put(index, win.details());
          }
        });
    return wins;
  }

  private static int firstWin(List<ReplayEvent> events) {
    for (int index = 0; index < events.size(); index++)
      if (events.get(index) instanceof Hora) return index;
    throw new AssertionError("The record has no win.");
  }
}
