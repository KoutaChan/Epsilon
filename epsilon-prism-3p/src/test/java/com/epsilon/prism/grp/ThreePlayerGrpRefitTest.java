package com.epsilon.prism.grp;

import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpTensor;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの GRP の学び直し（{@link GrpRefitter}）を確かめる。点数の順位の Φ から始め、三人打ちの自己対局の半荘の記録でゼロから学習し、
 * 候補のネットワークを三人打ちの形で書く。候補への切り替え（{@link PhiControl#switchTo}）では、三人打ちの Φ の項目を足す。
 */
public class ThreePlayerGrpRefitTest {

  private static final Variant VARIANT = Variant.TENHOU_3P;
  private static final BaseClass[] BASES = {
    BaseClass.B320, BaseClass.B640, BaseClass.B1280, BaseClass.MANGAN, BaseClass.HANEMAN
  };

  @Test(groups = "native", timeOut = 600_000)
  public void aScoreOnlyStartTrainsAThreePlayerCandidate() throws Exception {
    Path run = Files.createTempDirectory("prism-3p-refit");
    try {
      GrpFiles files = new GrpFiles(run);
      Files.writeString(run.resolve("state.json"), "{\"format\": 1, \"next\": 6}");
      PhiEntry start =
          new PhiEntry(PhiEntry.INITIAL_ID, PhiEntryKind.SCORE_RANK, VARIANT, "", "", "", "", "", "", 1, "", "test");
      PhiHistory.write(files.phiHistory(), PhiHistory.LINES.toJson(start));
      for (int t = 1; t <= 5; t++) {
        record(files, t, start, games(1_000L * t, 30, t));
      }
      RefitRecord record = GrpRefitter.refit(files, setup(), 5, RefitCause.SCHEDULE, () -> false).orElseThrow();
      Assert.assertEquals(record.status(), RefitStatus.TRAINED, record.detail());
      Assert.assertEquals(record.init(), GrpInit.SCRATCH);
      Path candidate = files.candidate(record.id());
      Assert.assertEquals(record.path(), candidate.toAbsolutePath().normalize().toString());
      GrpNetwork network = (GrpNetwork) GrpCheckpoint.read(candidate);
      Assert.assertEquals(network.shape(), GrpRefitter.shape(VARIANT));
      Assert.assertEquals(network.shape().variant(), VARIANT);
      PhiEntry entry = record.entry(VARIANT, 7, "test");
      Assert.assertEquals(entry.kind(), PhiEntryKind.NETWORK);
      Assert.assertEquals(entry.variant(), VARIANT);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** 三人打ちの run で READY の候補に切り替えると、足した Φ の項目は最初の項目と同じく三人打ちになる。 */
  @Test
  public void aReadyCandidateSwitchesInAsAThreePlayerEntry() throws Exception {
    Path run = Files.createTempDirectory("prism-3p-switch");
    try {
      GrpFiles files = new GrpFiles(run);
      ThreePlayerSettings.create(
          run,
          List.of(),
          "phi.source=SELF_GRP",
          "phi.start.kind=SCORE_RANK",
          "phi.grp.monitor=RECORD",
          "phi.grp.refit=MANUAL",
          "phi.grp.switching=PROPOSE");
      Files.writeString(run.resolve("state.json"), "{\"format\": 1, \"next\": 8}");
      Assert.assertEquals(PhiHistory.initialize(files, setup()).variant(), VARIANT);
      GrpShape shape = GrpRefitter.shape(VARIANT);
      float[] weights = new float[GrpTensor.starts(shape)[GrpTensor.values().length]];
      SplittableRandom random = new SplittableRandom(7);
      for (int index = 0; index < weights.length; index++) {
        weights[index] = (float) (random.nextGaussian() * 0.1);
      }
      Path candidate = run.resolve("candidate").toAbsolutePath().normalize();
      GrpCheckpoint.write(GrpNetwork.of(shape, weights), candidate);
      new RefitRecord(
              "g007-a1",
              7,
              RefitStatus.READY,
              RefitCause.MANUAL,
              new RecencyWindow(7, 2, 64),
              GrpInit.SCRATCH,
              PhiEntry.INITIAL_ID,
              candidate.toString(),
              "",
              "",
              null,
              null,
              "",
              1.0)
          .append(files);
      PhiEntry switched = PhiControl.switchTo(run, "g007-a1", null, "test").orElseThrow();
      Assert.assertEquals(switched.kind(), PhiEntryKind.NETWORK);
      Assert.assertEquals(switched.variant(), VARIANT);
      Assert.assertEquals(switched.effectiveFrom(), 8);
      Assert.assertEquals(PhiHistory.read(files).getLast(), switched);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  private static PhiSetup setup() {
    PrismSettings settings =
        ThreePlayerSettings.of(
            "phi.source=SELF_GRP",
            "phi.start.kind=SCORE_RANK",
            "phi.grp.monitor=RECORD",
            "phi.grp.trustGames=10",
            "phi.grp.threads=1");
    return new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed());
  }

  private static void record(GrpFiles files, int t, PhiEntry entry, List<GameRecord> games) throws IOException {
    Files.createDirectories(files.cohort(t));
    Files.writeString(files.cohort(t).resolve(CohortWriter.MANIFEST), "{}");
    Files.createDirectories(files.records(t).getParent());
    GameRecordCodec.write(files.records(t), new GameCohort(VARIANT, t, entry.id(), games));
  }

  /** 三人打ちのルールで点の精算と局の進行をした合成の半荘。局ごとに、荒牌・ロン・ツモを乱数で選ぶ。 */
  private static List<GameRecord> games(long first, int count, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    List<GameRecord> games = new ArrayList<>(count);
    for (int game = 0; game < count; game++) {
      games.add(game(first + game, random));
    }
    return games;
  }

  private static GameRecord game(long gameId, SplittableRandom random) {
    TableState table = new TableState(VARIANT);
    TableState next = new TableState(VARIANT);
    RoundOutcome outcome = new RoundOutcome(VARIANT);
    int[] delta = new int[3];
    List<GrpState> handEnds = new ArrayList<>();
    while (true) {
      int kind = random.nextInt(10);
      int winner = random.nextInt(3);
      int base = BASES[random.nextInt(BASES.length)].base();
      if (kind < 2) {
        outcome.setExhaustive(random.nextInt(8), 0, 0);
      } else if (kind < 6) {
        outcome.setRon(base, winner, (winner + 1 + random.nextInt(2)) % 3, RoundOutcome.NO_SEAT, 0);
      } else {
        outcome.setTsumo(base, winner, RoundOutcome.NO_SEAT, 0);
      }
      Settlement.apply(VARIANT, table, outcome, delta);
      Progression.next(table, outcome, delta, next);
      if (next.gameOver()) {
        int[] scores = new int[3];
        for (int seat = 0; seat < 3; seat++) {
          scores[seat] = next.score(seat);
        }
        return new GameRecord(VARIANT, gameId, handEnds, scores);
      }
      handEnds.add(GrpState.of(next));
      table.copyFrom(next);
    }
  }
}
