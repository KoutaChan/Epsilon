package com.epsilon.prism.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RandomGames;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.prism.Limits;
import com.epsilon.prism.TableWalk;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HexFormat;
import java.util.List;
import org.testng.Assert;

/** 同じ半荘の異なる時点から複数の判断を取り、入力の各領域に行を詰める。 */
public final class InputSamples {
  private InputSamples() {}

  public static final BatchPlan.Capacity CAPACITY =
      new BatchPlan.Capacity(
          12,
          12 * Limits.MAX_EVENT_TOKENS,
          12 * Limits.MAX_TRANSITIONS,
          12 * Limits.MAX_LEGAL_ACTIONS);

  public static void assertBaseBytes(InputFormat format, String expected)
      throws NoSuchAlgorithmException {
    HostBatch batch = encode(HostInputs.allocate(format, CAPACITY, ByteBuffer::allocateDirect));
    var digest = MessageDigest.getInstance("SHA-256");
    digest.update(batch.blob().slice(0, batch.bytes()));
    Assert.assertEquals(HexFormat.of().formatHex(digest.digest()), expected);
  }

  public static HostBatch encode(HostInputs batch) {
    var plan = batch.plan();
    var format = plan.format();
    var records = sample(format, plan.capacity().rows());
    var o = new Observation(format.variant());
    plan.clear();
    for (int row = 0; row < records.size(); row++)
      Assert.assertEquals(plan.add(o.bind(records, row)), row);
    plan.seal();
    var writer = format.writer();
    for (int row = 0; row < plan.rows(); row++) writer.write(o.bind(records, row), batch, row);
    return batch.input();
  }

  /** 長い河・副露・立直・槓・抜き北を含む判断の入力と、形式の記述を照合する。 */
  public static void assertHistory(InputFormat base, String description, Observation... scenes)
      throws Exception {
    var variant = base.variant();
    var rows = new ObservationBatch(variant, 4096);
    var single = new ObservationBatch(variant, 256);
    var capture = new ObservationWriter(variant);
    var o = new Observation(variant);
    var kinds = EnumSet.noneOf(EngineDecisionKind.class);
    int[] seen = {0};
    for (var policy : List.of(TableWalk.Policy.CALLS, TableWalk.Policy.GREEDY)) {
      new TableWalk(variant, policy)
          .run(
              7000 + policy.ordinal(),
              50,
              (engine, slot) -> {
                var decisions = engine.decisions();
                var kind = decisions.kind(slot);
                if (++seen[0] % 113 != 0 && kinds.contains(kind)) return;
                kinds.add(kind);
                single.clear();
                capture.capture(engine.view(decisions.seat(slot)), decisions, slot, single);
                rows.add(o.bind(single, 0));
              });
    }
    for (Observation scene : scenes) {
      rows.add(scene);
      kinds.add(scene.decisionKind());
    }
    for (var kind : EngineDecisionKind.values()) {
      if (kind.in(variant)) Assert.assertTrue(kinds.contains(kind), variant + " missing " + kind);
    }
    boolean longRiver = false, riichi = false, kan = false, nuki = false;
    for (int i = 0; i < rows.size(); i++) {
      o.bind(rows, i);
      int[] discards = new int[variant.seats()];
      for (int e = 0; e < o.events(); e++) {
        if (o.eventKind(e) == EventKind.DISCARD) longRiver |= ++discards[o.eventAuthor(e)] >= 16;
      }
      for (int seat = 0; seat < variant.seats(); seat++) {
        riichi |= o.riichi(seat) != RiichiState.NONE;
        nuki |= o.nuki(seat) > 0;
      }
      kan |= o.kanDoraIndicators() > 0;
    }
    Assert.assertTrue(longRiver && riichi && kan && (!variant.has(RuleFlag.NUKI) || nuki));
    List<String> reference = new ArrayList<>();
    reference.add(description);
    reference.add("rows=" + rows.size());
    for (boolean segments : new boolean[] {false, true}) {
      for (boolean causal : new boolean[] {false, true}) {
        var features = java.util.EnumSet.noneOf(InputFeature.class);
        if (segments) features.add(InputFeature.CANDIDATE_SEGMENTS);
        if (causal) features.add(InputFeature.PUBLIC_ORDER_EVENTS);
        var format = InputFormat.of(base.variant(), features);
        var host = HostInputs.allocate(format, CAPACITY, ByteBuffer::allocate);
        var writer = format.writer();
        var sha = MessageDigest.getInstance("SHA-256");
        for (int start = 0; start < rows.size(); start += CAPACITY.rows()) {
          int count = Math.min(rows.size() - start, CAPACITY.rows());
          var plan = host.plan();
          plan.clear();
          for (int i = 0; i < count; i++) Assert.assertEquals(plan.add(o.bind(rows, start + i)), i);
          plan.seal();
          for (int i = 0; i < count; i++) writer.write(o.bind(rows, start + i), host, i);
          sha.update(host.input().blob().slice(0, host.input().bytes()));
        }
        reference.add(
            "segments="
                + segments
                + " causal="
                + causal
                + " "
                + HexFormat.of().formatHex(sha.digest()));
      }
    }
    String name = variant + "-input.txt";
    try (var stream = InputSamples.class.getResourceAsStream(name)) {
      Assert.assertNotNull(stream, name);
      Assert.assertEquals(
          reference, new String(stream.readAllBytes(), StandardCharsets.UTF_8).lines().toList());
    }
  }

  private static ObservationBatch sample(InputFormat format, int rows) {
    var variant = format.variant();
    var game = RandomGames.start(variant, 721);
    var records = new ObservationBatch(variant, 4096);
    var capture = new ObservationWriter(variant);
    for (int step = 0; records.size() < rows; step++, game.step()) {
      if (game.boundary() != Boundary.DECISIONS || step % 5 != 0) continue;
      var decisions = game.engine.decisions();
      capture.capture(game.engine.view(decisions.seat(0)), decisions, 0, records);
    }
    return records;
  }
}
