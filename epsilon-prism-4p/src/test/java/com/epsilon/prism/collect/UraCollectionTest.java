package com.epsilon.prism.collect;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.luck.UraAudit;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.run.CollectReport;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 収集の局末の値と裏ドラの監査を確かめる（点数のなめらかな関数の Φ、立直と和了を選ぶ方策、帳簿つき）。監査の割合を 0・0.05・1 にしても、
 * ワーカーの数を変えても、行はビットまで一致する。主経路の局末の値は、局の後の卓から直接求めた値（続く局なら次局の卓の Φ、終局なら最終
 * 順位の効用）と一致する。
 * 割合 0 では期待値を求めない。割合 1 では全ての立直の和了で求めるので、期待値で学ぶ収集と同じ回数だけ Φ を呼ぶ。監査の区分ごとの和は
 * ワーカーの数に依らない。収集の段の報告は監査の要約を持ち、JSON から読み戻すと同じ報告になる。
 */
public class UraCollectionTest {

  private static final long SEED = 7_001L;
  private static final RankUtility UTILITY = RankUtility.TENHOU;

  /** ラベル行、ロールイン、写しの中の行がどれも出るようにした抽出率（{@code CollectorTest} と同じ）。 */
  private static final Sampling SAMPLING =
      TestSampling.grouped(0.25, 0.03, 0.03, 0.03, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  /** 一様分布を ε_b = 0.05 だけ混ぜる行動方策。 */
  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE);

  /** 収集のテストのハッシュの方策と同じ s に、和了なら 4、立直の打牌なら 2 を足し、σ をその argmax に置く方策（立直の和了を多く作るため）。 */
  static final class RiichiPolicy implements PolicyService {
    @Override
    public int version() {
      return 1;
    }

    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(PolicyRequest request) {
      int n = request.legalCount();
      float[] scores = new float[n];
      int best = 0;
      for (int a = 0; a < n; a++) {
        long bits = SeedMixer.mix64(request.legal()[a] * 31L + request.seat());
        float boost =
            switch (Action.of(request.legal()[a]).type()) {
              case TSUMO_AGARI, RON_AGARI -> 4f;
              case RIICHI_DAHAI -> 2f;
              default -> 0f;
            };
        scores[a] = boost + (float) CrnSeeds.unit(bits);
        if (scores[a] > scores[best]) best = a;
      }
      float[] sigma = new float[n];
      sigma[best] = 1f;
      request.scores = scores;
      request.sigma = sigma;
      if (request.retain()) {
        float[] prior = new float[n];
        double sum = 0;
        for (int a = 0; a < n; a++) {
          prior[a] = 1f + (float) CrnSeeds.unit(SeedMixer.mix64(request.legal()[a] * 17L + 5));
          sum += prior[a];
        }
        for (int a = 0; a < n; a++) prior[a] = (float) (prior[a] / sum);
        request.prior = prior;
      }
    }
  }

  /**
   * 主経路の局ごとに、局の後の卓から直接求めた局末の値（このクラスが自分の Φ で求める）。キーは半荘・局・本場。
   *
   * @param values 絶対席ごとの局末の値
   * @param terminal 半荘が終わった局なら true
   */
  record Former(double[] values, boolean terminal) {}

  /** 主経路を外から見て、局末ごとに、局の後の卓から直接その値を求める（進行のワーカーから並行に呼ばれる）。 */
  static final class FormerValues implements GameObserver {
    private final Phi phi = Phi.scoreSmooth(TENHOU_4P, UTILITY);
    final Map<String, Former> hands = new ConcurrentHashMap<>();

    @Override
    public void gameEnded(GameKey game, TableState last) {}

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      TableState next = end.next();
      double[] values = new double[TENHOU_4P.seats()];
      if (next.gameOver()) {
        for (int seat = 0; seat < values.length; seat++) values[seat] = UTILITY.value(TENHOU_4P, next.rank(seat));
      } else {
        phi.values(null, next, values);
      }
      SeatView view = full.seat(0);
      Assert.assertNull(hands.put(key(game.wall(), view.round(), view.honba()), new Former(values, next.gameOver())));
    }

    static String key(long gameId, int round, int honba) {
      return gameId + ":" + round + ":" + honba;
    }
  }

  /**
   * 1 つの収集の結果。
   *
   * @param result 収集の結果
   * @param rows 確定して読み戻した行
   * @param encoded 行の正準形（{@code CollectorTest.encoded}）
   * @param phiRequests 収集で Φ を呼んだ回数
   */
  record Collected(Collector.Result result, List<Row> rows, List<String> encoded, long phiRequests) {}

  /** 半荘 600〜611 を、帳簿つきで、点数のなめらかな関数の Φ と {@link RiichiPolicy} で集める。 */
  private static Collected collect(UraSettings ura, int workers, GameObserver observer) {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 12; id++) ids.add(600 + id);
    Phi phi = Phi.scoreSmooth(TENHOU_4P, UTILITY);
    try (var table = new TableLoop(new TableSettings(4, workers, 1000L, 4));
        var temporary = new CollectorTest.TemporaryCohort(CollectorTest.LEDGER)) {
      var cohort =
          new Collector.Cohort(
              TENHOU_4P, SEED, SEED * 17, ids, DrawOrder.SEAT_STREAMS, SAMPLING, CollectorTest.LEDGER, ura, FLOOR,
              TwinDiscard.TSUMOGIRI, observer);
      var result = Collector.collect(table, cohort, new RiichiPolicy(), phi, temporary.sink);
      List<Row> rows = temporary.sealAndRead();
      return new Collected(result, rows, CollectorTest.encoded(rows), phi.stats().requests());
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  @Test(timeOut = 1_800_000)
  public void theAuditLeavesTheRowsAndTheFormerValuesAlone() {
    FormerValues former = new FormerValues();
    Collected none = collect(new UraSettings.Realized(0), 4, former);
    Collected sampled = collect(new UraSettings.Realized(0.05), 4, null);
    Collected all = collect(new UraSettings.Realized(1), 4, null);
    Collected serial = collect(new UraSettings.Realized(1), 1, null);
    Collected expected = collect(new UraSettings.Expected(), 4, null);
    Assert.assertFalse(none.rows().isEmpty());
    Assert.assertEquals(sampled.encoded(), none.encoded(), "sample 0.05");
    Assert.assertEquals(all.encoded(), none.encoded(), "sample 1");
    Assert.assertEquals(serial.encoded(), none.encoded(), "sample 1, one worker");
    Assert.assertNotEquals(expected.encoded(), none.encoded(), "learning the expected ura changed no row");

    // 主経路の（行, 手）（MAIN と ROOT の打った手）の Y は、局の後の卓から直接求めた局末の値を float にしたもの（終局も同じ）
    Observation observation = new Observation(TENHOU_4P);
    int compared = 0, terminal = 0;
    for (Row row : none.rows()) {
      if (row.source() != Row.Source.MAIN && row.source() != Row.Source.ROOT) continue;
      observation.bind(row.observation(), 0);
      Former hand = former.hands.get(FormerValues.key(row.gameId(), observation.kyoku(), observation.honba()));
      Assert.assertNotNull(hand, row.gameId() + " " + observation.kyoku() + " " + observation.honba());
      float y = (float) hand.values()[observation.observer()];
      Assert.assertEquals(
          Float.floatToRawIntBits(row.values()[row.played()]), Float.floatToRawIntBits(y), row.source().toString());
      compared++;
      if (hand.terminal()) terminal++;
    }
    Assert.assertTrue(compared > 100 && terminal > 0, compared + " compared, " + terminal + " terminal");

    UraAudit.Report off = none.result().ura(), on = all.result().ura(), few = sampled.result().ura();
    String counts = "off " + off.summary() + " on " + on.summary();
    Assert.assertTrue(off.handEnds() > 0 && off.riichiWins() > 0, counts);
    for (UraAudit.Report report : List.of(few, on, serial.result().ura(), expected.result().ura())) {
      Assert.assertEquals(report.handEnds(), off.handEnds(), counts);
      Assert.assertEquals(report.riichiWins(), off.riichiWins(), counts);
    }
    Assert.assertEquals(off.averaged(), 0);
    Assert.assertTrue(off.cells().isEmpty());
    Assert.assertEquals(off.microsPerAveraged(), 0.0);
    Assert.assertTrue(few.averaged() <= off.riichiWins(), counts);
    Assert.assertEquals(on.averaged(), on.riichiWins(), counts);
    Assert.assertEquals(expected.result().ura().averaged(), on.averaged(), counts);
    long binned = 0;
    for (long bin : on.groups()) binned += bin;
    Assert.assertEquals(binned, on.averaged(), counts);
    Assert.assertFalse(on.cells().isEmpty(), counts);
    Assert.assertEquals(serial.result().ura().cells(), on.cells(), "cells by workers");
    Assert.assertEquals(serial.result().ura().summary().cells(), on.summary().cells(), "cells by workers");
    // 標準誤差は束が 2 つあれば求まる。要約の z は、束が足りる区分にだけ書く（この小さな収集では、どの区分も束が少ない）
    Assert.assertTrue(on.cells().stream().anyMatch(cell -> cell.standardError() > 0), counts);
    for (int index = 0; index < on.cells().size(); index++) {
      UraAudit.Cell cell = on.cells().get(index);
      boolean enough = cell.clusters() >= UraAudit.MIN_CLUSTERS && cell.standardError() > 0;
      Assert.assertEquals(on.summary().cells().get(index).z() != null, enough, cell.toString());
    }
    // 監査は Φ を呼ぶ回数を増やすだけ。全ての立直の和了で期待値を求めると、回数は期待値で学ぶ収集と同じになる
    Assert.assertTrue(all.phiRequests() > none.phiRequests(), all.phiRequests() + " vs " + none.phiRequests());
    Assert.assertEquals(expected.phiRequests(), all.phiRequests());
    Assert.assertEquals(serial.phiRequests(), all.phiRequests());

    CollectReport report = CollectReport.of(1, all.result(), all.rows().size(), 0);
    Assert.assertEquals(report.ura(), on.summary());
    JsonObject json = JsonParser.parseString(RecordFile.json(report)).getAsJsonObject().getAsJsonObject("ura");
    Assert.assertEquals(json.get("handEnds").getAsLong(), on.handEnds());
    Assert.assertEquals(json.get("riichiWins").getAsLong(), on.riichiWins());
    Assert.assertEquals(json.get("averaged").getAsLong(), on.averaged());
    Assert.assertEquals(json.getAsJsonArray("groups").size(), UraAudit.GROUP_BINS);
    Assert.assertEquals(json.getAsJsonArray("cells").size(), on.cells().size());
    Assert.assertEquals(RecordFile.read(CollectReport.class, RecordFile.json(report), "test").value(), report);
  }
}
