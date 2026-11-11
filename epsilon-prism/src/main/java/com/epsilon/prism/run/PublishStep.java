package com.epsilon.prism.run;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.settings.MemorySettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 公開の段。方策の記憶を使わなければ、生徒 θ_{t+1} のファイルをそのまま写してバージョン t + 1 にする。使うなら、打つネットワークを
 * A_{t+1} = (1 − η)·A_start + η·θ_{t+1}（{@link MemorySettings#step}、CPU・FP32）にし、θ_{t+1} を iterate/ に添える。
 * snapshots/&lt;t+1&gt;.part に書いて中身ごと fsync し、行き先が無いときだけ snapshots/&lt;t+1&gt; に rename する。この rename が
 * 完了マーカーになる。段の報告は work の {@value #FILE}。バージョンがもう完成していれば作り直さず、報告だけを書く。
 */
final class PublishStep implements Step<PrismIteration> {

  /** 段の報告のファイル（work の中）。 */
  static final String FILE = "PUBLISH.json";

  private static final Logger LOG = LoggerFactory.getLogger(PublishStep.class);

  @Override
  public Optional<StepReport> done(PrismIteration iteration) throws IOException {
    Path snapshot = iteration.run().snapshot(iteration.version());
    if (!Files.isDirectory(snapshot)) {
      return Optional.empty();
    }
    Path file = iteration.work().resolve(FILE);
    if (Files.exists(file)) {
      return Optional.of(StepReport.read(file, PublishReport.class));
    }
    StepReport report = StepReport.ran(iteration, System.nanoTime(), new PublishReport(iteration.version(), average(snapshot)));
    report.write(file);
    return Optional.of(report);
  }

  @Override
  public StepReport run(PrismIteration iteration) throws Exception {
    long started = System.nanoTime();
    RunDirectory run = iteration.run();
    int version = iteration.version();
    Path target = run.snapshot(version);
    Path part = PrismStep.part(target);
    DurableFiles.delete(part);
    Path learned = iteration.work().resolve(LearnStep.LEARN).resolve(LearnStep.STUDENT);
    MemorySettings memory = iteration.settings().memory();
    PolicyParams params = PolicyParams.of(version);
    Snapshot.AverageRecord record = null;
    if (!memory.averages()) {
      Snapshot.fillCopy(part, learned, params);
    } else {
      MemorySettings.Step step = memory.step(version);
      record = new Snapshot.AverageRecord(memory.scheme(), step.eta(), memory.from());
      try (NDManager cpu = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
        PrismNetwork average = PrismNetwork.load(Snapshot.model(run.snapshot(iteration.start())), cpu);
        average.mixIn(PrismNetwork.load(learned, cpu), step.eta());
        Snapshot.fillAveraged(part, learned, average, record, params);
      }
      LOG.info(
          "t={} averaged version {}: scheme={} eta={}",
          iteration.number(),
          version,
          memory.scheme(),
          String.format("%.4f", step.eta()));
    }
    DurableFiles.seal(part, target);
    StepReport report = StepReport.ran(iteration, started, new PublishReport(version, record));
    report.write(iteration.work().resolve(FILE));
    return report;
  }

  /** 完成したバージョンから方策の記憶の記録を読む（記憶を使わなければ null）。 */
  private static Snapshot.AverageRecord average(Path snapshot) throws IOException {
    Path file = snapshot.resolve(Snapshot.AVERAGE);
    return Files.exists(file) ? RecordFile.read(Snapshot.AverageRecord.class, file).value() : null;
  }
}
