package com.epsilon.reviewer.engine;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.ClassicFamily;
import com.epsilon.classic.RandomCheckpoints;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.io.RecordFiles;
import com.epsilon.mahjong.io.majsoul.MajSoulReaderTest;
import com.epsilon.mahjong.io.mjai.MjaiCodec;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.FixtureFamily;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.engine.review.Analysis;
import com.epsilon.reviewer.engine.review.JobControl;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 牌譜の読み込み（天鳳・mjai・雀魂の形式と、断る牌譜）、候補の並べ替え、取り消しと進み具合、ランダムな重みで作った classic の
 * チェックポイントでの解析を確かめる。
 */
public class AnalysisTest {
  /** 解析する、エンジンで打った四人打ちの半荘の乱数の種。 */
  private static final long SEED = 0;

  @Test
  public void tenhouRecordsReadGzipBomAndUrlEncodedNames() {
    LoadedRecord record = played();
    RecordedGame game = record.game();
    Assert.assertEquals(record.summary().source(), "tenhou");
    Assert.assertEquals(record.summary().roundCount(), game.rounds().size());
    Assert.assertEquals(
        record.summary().finalScores(), Arrays.stream(game.finalScores()).boxed().toList());
    String xml =
        gunzip(TestFiles.played(Variant.TENHOU_4P, SEED))
            .replaceFirst(" n0=\"[^\"]*\"", " n0=\"%E3%81%BF%E3%81%AA%E3%81%A8\"");
    LoadedRecord renamed = RecordLoader.load(("\uFEFF" + xml).getBytes(StandardCharsets.UTF_8));
    Assert.assertEquals(renamed.summary().names().getFirst(), "みなと");
    Assert.assertEquals(renamed.summary().names().subList(1, 4), record.summary().names().subList(1, 4));
    Assert.assertEquals(renamed.rounds(), record.rounds());
  }

  @Test
  public void mjaiRecordsReadAsLinesOrAnArray() {
    LoadedRecord tenhou = played();
    String lines = MjaiCodec.write(tenhou.game());
    LoadedRecord record = load(lines);
    Assert.assertEquals(record.summary().source(), "mjai");
    Assert.assertEquals(record.summary().names(), tenhou.summary().names());
    Assert.assertEquals(record.summary().finalScores(), tenhou.summary().finalScores());
    Assert.assertEquals(record.summary().roundCount(), tenhou.summary().roundCount());
    JsonArray array = new JsonArray();
    for (String line : lines.split("\n")) if (!line.isBlank()) array.add(JsonParser.parseString(line));
    Assert.assertEquals(load(array.toString()).game(), record.game());
  }

  @Test
  public void decodedMahjongSoulRecordKeepsItsSource() {
    LoadedRecord record = load(MajSoulReaderTest.RECORD);
    Assert.assertEquals(record.summary().source(), "majsoul");
    Assert.assertEquals(record.summary().names(), List.of("East", "", "West", ""));
    Assert.assertEquals(record.summary().finalScores(), List.of(24000, 21000, 31000, 24000));
    Assert.assertEquals(record.summary().roundCount(), 1);
    Assert.assertEquals(record.firstStep(0), 0);
  }

  @Test
  public void recordsThatCannotBeReviewedAreRejected() {
    LoadedRecord record = played();
    List<String> lines = List.of(MjaiCodec.write(record.game()).split("\n"));
    rejected(Format6.text(record, new RecordingPolicy()), "An analysis result is not a record");
    rejected(new byte[RecordFiles.MAX_BYTES + 1], "16 MiB");
    rejected("[".repeat(RecordFiles.MAX_NESTING + 1) + "]".repeat(RecordFiles.MAX_NESTING + 1), "nests");
    int events = record.rounds().stream().mapToInt(round -> round.events().size()).sum();
    List<String> many = new ArrayList<>(List.of(lines.getFirst()));
    for (int copy = 0; copy <= RecordFiles.MAX_EVENTS / events; copy++)
      many.addAll(lines.subList(1, lines.size() - 1));
    many.add(lines.getLast());
    rejected(String.join("\n", many), "more than " + RecordFiles.MAX_EVENTS);
    rejected(String.join("\n", riichiBeforeTheFirstDiscard(lines)), "Record replay failed");
    rejected(String.join("\n", negatedFirstWin(lines)), "Record replay failed");
    rejected(String.join("\n", lines.subList(0, lines.size() - 1)), "ends before the game does");
  }

  @Test(groups = "native")
  public void randomClassicCheckpointReviewsEverySeatWithNormalizedProbabilities()
      throws Exception {
    Path directory = TestFiles.directory("reviewer-native-");
    Path checkpoint = directory.resolve("checkpoint");
    try {
      RandomCheckpoints.write(checkpoint, Architecture.NANO, Variant.TENHOU_4P, 7);
      try (Policy policy =
          new ClassicFamily.Nano()
              .open(
                  checkpoint,
                  Variant.TENHOU_4P,
                  PolicyOptions.defaults())) {
        Format6.Document document = Format6.analyze(played(), model(policy));
        Assert.assertFalse(document.json().has("temperature"), "Nano has no value head");
        Set<Integer> seats = new HashSet<>();
        for (Format6.Located located : document.decisions()) {
          Format6.Step step = located.step();
          seats.add(step.actor());
          double sum = 0;
          for (int i = 0; i < step.candidates().size(); i++) {
            if (i > 0) Assert.assertTrue(value(step, i - 1) >= value(step, i));
            sum += value(step, i);
          }
          Assert.assertEquals(sum, 1.0, 1e-4);
        }
        Assert.assertEquals(seats, Set.of(0, 1, 2, 3));
      }
    } finally {
      TestFiles.delete(directory);
    }
  }

  /**
   * 価値は行動の番号 2 つごとに同じ値にし、確率は番号が大きいほど高くする。候補は価値の損失が小さい順に並ぶ。損失が同じ候補は、モデルが
   * 貪欲に選ぶ手（確率の最も高い手）が先で、ほかは合法手の順のまま残る。牌譜で選んだ手は、並べ替えた後も選んだ手として残る。
   */
  @Test
  public void valuedPolicySortsByValueLossAndKeepsTheRecordedChoice() {
    LoadedRecord record = played();
    List<List<Integer>> legal = new ArrayList<>();
    Format6.Document document =
        Format6.analyze(
            record,
            fake(
                true,
                observation -> {
                  List<Integer> codes = codes(observation);
                  legal.add(codes);
                  float[] values = new float[codes.size()];
                  float[] logits = new float[codes.size()];
                  for (int i = 0; i < codes.size(); i++) {
                    values[i] = -0.001f * (codes.get(i) / 2);
                    logits[i] = 0.05f * codes.get(i);
                  }
                  return evaluation(softmax(logits), values);
                }));
    Assert.assertEquals(
        document.json().get("temperature").getAsDouble(), FixtureFamily.TEMPERATURE);
    List<Format6.Step> decisions = steps(document);
    List<Integer> recorded = recordedCodes(record);
    Assert.assertEquals(decisions.size(), recorded.size());
    Assert.assertEquals(legal.size(), recorded.size());
    Assert.assertTrue(decisions.stream().anyMatch(step -> step.candidates().size() > 2));
    for (int s = 0; s < decisions.size(); s++) {
      Format6.Step step = decisions.get(s);
      List<Integer> ids = new ArrayList<>();
      for (int i = 0; i < step.candidates().size(); i++) ids.add(step.actionId(i));
      int greedy = Collections.max(legal.get(s));
      List<Integer> expected =
          legal.get(s).stream()
              .sorted(Comparator.comparingInt((Integer id) -> id / 2).thenComparing(id -> id != greedy))
              .toList();
      Assert.assertEquals(ids, expected);
      int best = ids.getFirst() / 2;
      for (int i = 0; i < ids.size(); i++)
        Assert.assertEquals(value(step, i), 0.001f * (ids.get(i) / 2 - best), 1e-5f);
      int chosen = step.chosen() < 0 ? -1 : ids.get(step.chosen());
      Assert.assertEquals(chosen, recorded.get(s).intValue());
    }
  }

  /**
   * 価値が全て同じなら、モデルが貪欲に選ぶ手を先にし、ほかは合法手の順に並べる（PRISM は同値な打牌のうち切る方の手を貪欲にするので、
   * その手が先に来る）。
   */
  @Test
  public void equalCandidatesPutTheGreedyActionFirst() {
    List<List<Integer>> legal = new ArrayList<>();
    Format6.Document document =
        Format6.analyze(
            played(),
            fake(
                true,
                observation -> {
                  List<Integer> codes = codes(observation);
                  legal.add(codes);
                  int n = codes.size();
                  float[] probabilities = new float[n];
                  Arrays.fill(probabilities, 1f / n);
                  return new Evaluation(probabilities, new float[n], n - 1);
                }));
    List<Format6.Step> decisions = steps(document);
    Assert.assertEquals(decisions.size(), legal.size());
    Assert.assertTrue(decisions.stream().anyMatch(step -> step.candidates().size() > 2));
    for (int s = 0; s < decisions.size(); s++) {
      Format6.Step step = decisions.get(s);
      List<Integer> ids = new ArrayList<>();
      for (int i = 0; i < step.candidates().size(); i++) ids.add(step.actionId(i));
      List<Integer> codes = legal.get(s);
      List<Integer> expected = new ArrayList<>(codes.size());
      expected.add(codes.getLast());
      expected.addAll(codes.subList(0, codes.size() - 1));
      Assert.assertEquals(ids, expected);
    }
  }

  @Test
  public void probabilityOnlyPolicyHasNoTemperatureAndSortsByProbability() {
    Format6.Document document =
        Format6.analyze(
            played(),
            fake(
                false,
                observation -> {
                  List<Integer> codes = codes(observation);
                  float[] logits = new float[codes.size()];
                  for (int i = 0; i < codes.size(); i++) logits[i] = 0.05f * codes.get(i);
                  return evaluation(softmax(logits), null);
                }));
    Assert.assertFalse(document.json().has("temperature"));
    List<Format6.Step> decisions = steps(document);
    Assert.assertTrue(decisions.stream().anyMatch(step -> step.candidates().size() > 2));
    for (Format6.Step step : decisions) {
      for (int i = 1; i < step.candidates().size(); i++)
        Assert.assertTrue(value(step, i - 1) >= value(step, i));
    }
  }

  @Test
  public void policyOutputForAnotherNumberOfActionsIsRejected() {
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            Format6.analyze(
                played(),
                fake(
                    false,
                    observation -> {
                      float[] probabilities = new float[observation.legalActions() + 1];
                      Arrays.fill(probabilities, 1f / probabilities.length);
                      return evaluation(probabilities, null);
                    })));
  }

  @Test
  public void aModelOfAnotherVariantIsUnavailable() {
    ReviewModel threePlayer =
        new ReviewModel() {
          @Override
          public Evaluation evaluate(Observation observation) {
            throw new AssertionError("The model is not called.");
          }

          @Override
          public boolean valued() {
            return false;
          }

          @Override
          public double temperature() {
            return FixtureFamily.TEMPERATURE;
          }

          @Override
          public Variant variant() {
            return Variant.TENHOU_3P;
          }
        };
    ReviewerException refused =
        Assert.expectThrows(
            ReviewerException.class, () -> Format6.analyze(played(), threePlayer));
    Assert.assertEquals(refused.code(), ErrorCode.MODEL_UNAVAILABLE);
  }

  @Test
  public void cancellationStopsAtTheNextEventAndProgressStaysBelowTheEnd() throws Exception {
    LoadedRecord record = played();
    List<Integer> progress = new ArrayList<>();
    Assert.expectThrows(
        CancellationException.class,
        () -> run(record, new RecordingPolicy(), cancelledAfter(5, progress)));
    Assert.assertEquals(progress.size(), 5);
    List<Integer> complete = new ArrayList<>();
    run(record, new RecordingPolicy(), cancelledAfter(Integer.MAX_VALUE, complete));
    Assert.assertEquals(complete.getLast().intValue(), 99);
    for (int i = 1; i < complete.size(); i++)
      Assert.assertTrue(complete.get(i - 1) <= complete.get(i));
  }

  /** 解析する牌譜（エンジンで打った四人打ちの半荘）。 */
  private static LoadedRecord played() {
    return PlayedRecords.read(Variant.TENHOU_4P, SEED);
  }

  private static void rejected(String text, String message) {
    rejected(text.getBytes(StandardCharsets.UTF_8), message);
  }

  private static void rejected(byte[] bytes, String message) {
    IllegalArgumentException refused =
        Assert.expectThrows(IllegalArgumentException.class, () -> RecordLoader.load(bytes));
    Assert.assertTrue(refused.getMessage().contains(message), refused.getMessage());
  }

  /** 最初の局の最初の打牌の前に立直の宣言を、打牌の後に立直の成立を置く。手は配牌のままで聴牌していないので、この立直は合法でない。 */
  private static List<String> riichiBeforeTheFirstDiscard(List<String> lines) {
    List<String> edited = new ArrayList<>(lines);
    for (int index = 0; index < edited.size(); index++) {
      JsonObject event = JsonParser.parseString(edited.get(index)).getAsJsonObject();
      if (!event.get("type").getAsString().equals("dahai")) continue;
      int actor = event.get("actor").getAsInt();
      edited.add(index + 1, "{\"type\":\"reach_accepted\",\"actor\":" + actor + "}");
      edited.add(index, "{\"type\":\"reach\",\"actor\":" + actor + "}");
      return edited;
    }
    throw new AssertionError("The record has no discard.");
  }

  /** 最初の和了で動く点数の正負を逆にする。局末の点数が記録と合わなくなる。 */
  private static List<String> negatedFirstWin(List<String> lines) {
    List<String> edited = new ArrayList<>(lines);
    for (int index = 0; index < edited.size(); index++) {
      JsonObject event = JsonParser.parseString(edited.get(index)).getAsJsonObject();
      if (!event.get("type").getAsString().equals("hora")) continue;
      JsonArray negated = new JsonArray();
      for (var delta : event.getAsJsonArray("deltas")) negated.add(-delta.getAsInt());
      event.add("deltas", negated);
      edited.set(index, event.toString());
      return edited;
    }
    throw new AssertionError("The record has no win.");
  }

  private static LoadedRecord load(String text) {
    return RecordLoader.load(text.getBytes(StandardCharsets.UTF_8));
  }

  private static String gunzip(byte[] bytes) {
    try (var in = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
      return new String(in.readAllBytes(), StandardCharsets.UTF_8);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /**
   * 牌譜どおりにエンジンを進め、判断の時点ごとに、牌譜で選んだ手の行動の番号を、判断する席の順に並べる。牌譜に記録されていない応答は
   * -1 にする。
   */
  private static List<Integer> recordedCodes(LoadedRecord record) {
    var variant = record.game().variant();
    var driver = new ReplayDriver(variant);
    var ws = new RuleWorkspace(variant);
    List<Integer> codes = new ArrayList<>();
    for (RecordedRound round : record.rounds()) {
      Boundary boundary = driver.start(round, ws);
      while (boundary == Boundary.DECISIONS) {
        Decisions decisions = driver.engine().decisions();
        for (int slot = 0; slot < decisions.size(); slot++) {
          int choice = driver.choice(slot);
          codes.add(choice == ReplayDriver.UNOBSERVED ? -1 : decisions.legal(slot, choice));
        }
        boundary = driver.step(ws);
      }
    }
    return codes;
  }

  private static void run(LoadedRecord record, ReviewModel model, JobControl control)
      throws Exception {
    try (var writer =
        new ResultWriter(
            OutputStream.nullOutputStream(),
            "00000000-0000-4000-8000-000000000000",
            "2026-11-04T00:00:00Z",
            Format6.MODEL,
            record.summary())) {
      Analysis.run(record, model, control, writer);
    }
  }

  /** 進み具合を {@code count} 回知らされた後で取り消す。 */
  private static JobControl cancelledAfter(int count, List<Integer> progress) {
    return new JobControl() {
      @Override
      public boolean cancelled() {
        return progress.size() >= count;
      }

      @Override
      public void progress(int percent) {
        progress.add(percent);
      }
    };
  }

  private static List<Format6.Step> steps(Format6.Document document) {
    return document.decisions().stream().map(Format6.Located::step).toList();
  }

  private static float value(Format6.Step step, int candidate) {
    return step.candidates().get(candidate).getAsJsonArray().get(1).getAsFloat();
  }

  private static List<Integer> codes(Observation observation) {
    return IntStream.range(0, observation.legalActions())
        .mapToObj(i -> observation.action(i).code())
        .toList();
  }

  private static float[] softmax(float[] logits) {
    double max = Double.NEGATIVE_INFINITY;
    for (float logit : logits) max = Math.max(max, logit);
    double[] weights = new double[logits.length];
    double sum = 0;
    for (int i = 0; i < logits.length; i++) {
      weights[i] = Math.exp(logits[i] - max);
      sum += weights[i];
    }
    float[] probabilities = new float[logits.length];
    for (int i = 0; i < logits.length; i++) probabilities[i] = (float) (weights[i] / sum);
    return probabilities;
  }

  private static Evaluation evaluation(float[] probabilities, float[] values) {
    int greedy = 0;
    for (int i = 1; i < probabilities.length; i++)
      if (probabilities[i] > probabilities[greedy]) greedy = i;
    return new Evaluation(probabilities, values, greedy);
  }

  /** 観測の合法手だけを見て評価する、偽のモデル（四人打ち）。 */
  private static ReviewModel fake(boolean valued, Function<Observation, Evaluation> evaluator) {
    return new ReviewModel() {
      @Override
      public Evaluation evaluate(Observation observation) {
        return evaluator.apply(observation);
      }

      @Override
      public boolean valued() {
        return valued;
      }

      @Override
      public double temperature() {
        return FixtureFamily.TEMPERATURE;
      }

      @Override
      public Variant variant() {
        return Variant.TENHOU_4P;
      }
    };
  }

  private static ReviewModel model(Policy policy) {
    return fake(false, policy::evaluate);
  }
}
