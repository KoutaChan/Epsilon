package com.epsilon.prism.app;

import ai.djl.Model;
import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Span;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.backend.KernelDevices;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.kernel.KernelProvider;
import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.kernel.KernelProviders;
import com.epsilon.prism.learn.AnchorScores;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.learn.Teacher;
import com.epsilon.prism.luck.LedgerCorrection;
import com.epsilon.prism.luck.LedgerReport;
import com.epsilon.prism.luck.LuckSettlement;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.noise.TempoLrMode;
import com.epsilon.prism.noise.TempoSettings;
import com.epsilon.prism.policy.CollectionPlans;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Executors;

/** 複数のコホートを並べ替え、固定したモデルを基準に、同じ順列と学習率の進行を引き継いで 1 エポック学習する。 */
@Named(
    value = "fit-plateau",
    doc = "Fit one resumable pass over saved cohorts with the combined TEMPO gain")
public record FitPlateau(
    @Key(doc = "Run containing the sealed cohorts", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Inclusive cohort range", presence = Presence.REQUIRED) String cohorts,
    @Key(doc = "Frozen starting snapshot", presence = Presence.REQUIRED) Path start,
    @Key(doc = "Recorded TEMPO gains and preceding R2 for the range", presence = Presence.REQUIRED)
        Path schedule,
    @Key(doc = "Arm output directory", presence = Presence.REQUIRED) Path out,
    @Key(doc = "開始モデルに要求する形（省略時は保存した形を使う）", presence = Presence.ABSENT) ShapeArm shape,
    @Key(
            doc = "Combined teacher gain (default: product of the frozen schedule)",
            presence = Presence.ABSENT)
        @Range(min = 0, minInclusive = false, max = 1)
        Double gain,
    @Key(
            doc =
                "Optional multiplier for both lineage learning rates; the teacher and Adam epsilon"
                    + " keep their gain",
            presence = Presence.ABSENT)
        @Range(min = 0, minInclusive = false)
        Double lrScale,
    @Key(
            doc = "Require the native kernels for scoring the anchor and training the student",
            value = "false")
        boolean requireNative,
    @Key(doc = "Maximum optimizer steps per invocation", value = "512") @Range(min = 1)
        int chunkSteps,
    @Key(doc = "Shuffle replica", value = "0") @Range(min = 0) int replicate)
    implements PrismCommand {

  public record Schedule(
      int first,
      List<Double> priorR2,
      List<Double> gains,
      double gain,
      double lrPeak,
      double lrFloor) {
    double combined(Span span, TempoSettings tempo) {
      if (!tempo.anchored()
          || first != span.first()
          || gains.size() != span.last() - first + 1
          || priorR2.size() != gains.size())
        throw new IllegalArgumentException("The TEMPO history must cover the requested iterations");
      double keep = 1;
      for (int i = 0; i < gains.size(); i++) {
        double r2 = priorR2.get(i);
        double k = tempo.gain(first + i, r2);
        if (!(r2 >= 0 && r2 <= 1 && k > 0 && k <= 1 && Math.abs(k - gains.get(i)) < 1e-12))
          throw new IllegalArgumentException(
              "TEMPO history differs from the run at t=" + (first + i));
        keep *= 1 - k;
      }
      if (!(Math.abs(1 - keep - gain) < 1e-12))
        throw new IllegalArgumentException("Invalid cumulative TEMPO gain");
      return 1 - keep;
    }
  }

  record Update(double scheduledGain, double teacherGain, double lrScale) {
    static Update of(
        double scheduled, Double requested, Double requestedLrScale, TempoSettings settings) {
      if (!(settings instanceof TempoSettings.On tempo) || tempo.lrMode() != TempoLrMode.GAIN) {
        throw new IllegalArgumentException(
            "The plateau fit requires the lineage TEMPO gain-based learning rate");
      }
      double chosen = requested == null ? scheduled : requested;
      if (!(chosen > 0 && chosen <= 1))
        throw new IllegalArgumentException("gain must be in (0, 1]");
      double scale =
          requestedLrScale == null
              ? tempo.lrMode().scale(chosen, tempo.lrReferenceGain())
              : requestedLrScale;
      if (!(Double.isFinite(scale) && scale > 0))
        throw new IllegalArgumentException("lr-scale must be finite and positive");
      return new Update(scheduled, chosen, scale);
    }
  }

  private record Recipe(
      String run,
      String start,
      String inputs,
      List<String> overrides,
      String shape,
      int seedReplica,
      int first,
      int last,
      Update update) {}

  private static final Gson FIT_JSON =
      new GsonBuilder().serializeSpecialFloatingPointValues().create();

  /**
   * 再開のたびに照合する、アンカーの採点と学習の計算。
   *
   * @param devices 学習のデバイス
   * @param anchorInference アンカーの採点に使う計算（{@link KernelDevices#label}）
   * @param nativeTrainingAttention 学習の Attention を専用カーネルで計算するか
   */
  private record Execution(
      List<String> devices, String anchorInference, boolean nativeTrainingAttention) {}

  private record Progress(int steps, long rows, double elapsedSeconds) {}

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    var span = Span.of("cohorts", cohorts);
    var settings = FitCohorts.settings(run, context.layers()).value();
    var history = RecordFile.read(Schedule.class, schedule).value();
    var tempo = settings.noise().tempo();
    var update = Update.of(history.combined(span, tempo), gain, lrScale, tempo);
    if (settings.fit().lrPeak() != history.lrPeak()
        || settings.fit().lrFloor() != history.lrFloor())
      throw new IllegalArgumentException("This arm requires the recorded lineage learning rate");
    Path anchor = Snapshot.model(start);
    var header = PrismNetwork.readHeader(anchor);
    var original = header.shape();
    var student = shape == null ? original : shape.shape();
    if (!original.equals(student))
      throw new IllegalArgumentException("Starting shape mismatch: " + original + " -> " + student);
    if (PrismNetwork.readFormat(anchor).variant() != settings.run().variant())
      throw new IllegalArgumentException("The start model does not match the run variant");
    settings.runtime().apply();
    settings.diagnostics().apply();
    Files.createDirectories(out);
    try (var channel =
            FileChannel.open(
                out.resolve("fit.lock"), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
        var lock = channel.tryLock()) {
      if (lock == null) throw new IOException("Another fitter owns " + out);
      var digest = MessageDigest.getInstance("SHA-256");
      for (Path path :
          List.of(
              schedule,
              run.resolve("settings.jsonl"),
              anchor.resolve("model.json"),
              anchor.resolve("weights.bin"))) digest.update(Files.readAllBytes(path));
      for (int t = span.first(); t <= span.last(); t++)
        digest.update(
            Files.readAllBytes(CohortWriter.directory(run, t).resolve(CohortWriter.MANIFEST)));
      var recipe =
          new Recipe(
              run.toRealPath().toString(),
              start.toRealPath().toString(),
              HexFormat.of().formatHex(digest.digest()),
              context.layers().stream().map(layer -> Json.write(layer.tree())).toList(),
              shape == null ? "START" : shape.name(),
              replicate,
              span.first(),
              span.last(),
              update);
      Path file = out.resolve("recipe.json");
      if (Files.exists(file)) {
        if (!recipe.equals(RecordFile.read(Recipe.class, file).value()))
          throw new IOException("Arm inputs changed: " + out);
      } else {
        DurableFiles.write(file, recipe);
      }
      var devices = settings.learner().networkDevices();
      Optional<Backend> scoring = Optional.empty();
      boolean nativeTraining = false;
      if (devices.primary().isGpu()) {
        if (settings.inference().kernels() == Kernels.NATIVE) {
          var teacher = new PrismNetwork(header.format(), original);
          scoring =
              CollectionPlans.select(
                      teacher,
                      List.of(devices.primary()),
                      Precision.BFLOAT16,
                      BatchPlan.Capacity.standard(teacher.format()),
                      null)
                  .map(CollectionPlans::backend);
        }
        var network = new PrismNetwork(InputFormat.of(header.format().variant()), student);
        var request =
            KernelDevices.request(
                network,
                devices.primary(),
                BatchPlan.Capacity.standard(network.format()),
                KernelProvider.Use.TRAINING);
        nativeTraining = KernelProviders.training(request, KernelProviders.discover()).isPresent();
      }
      var execution =
          new Execution(
              devices.devices().stream().map(Object::toString).toList(),
              KernelDevices.label(scoring),
              nativeTraining);
      if (requireNative && (scoring.isEmpty() || !nativeTraining)) {
        throw new IllegalArgumentException(
            "Expected native scoring and training for the plateau arm");
      }
      Path executionFile = out.resolve("execution.json");
      if (Files.exists(executionFile)) {
        if (!execution.equals(RecordFile.read(Execution.class, executionFile).value())) {
          throw new IOException("The execution backend changed while resuming " + out);
        }
      } else {
        if (requireNative && (Files.exists(out.resolve("snapshot")) || lastCheckpoint() != null)) {
          throw new IOException("The existing fit did not record its execution backend");
        }
        DurableFiles.write(executionFile, execution);
      }
      if (Files.exists(out.resolve("snapshot"))) {
        DurableFiles.write(
            out.resolve("progress.json"),
            Files.readAllBytes(out.resolve("snapshot/progress.json")));
        return Outcome.ok("Completed " + out);
      }
      return fit(settings, span, anchor, student, update);
    }
  }

  private Outcome fit(PrismSettings s, Span span, Path anchor, PrismShape student, Update update)
      throws IOException {
    double gain = update.teacherGain();
    var devices = s.learner().networkDevices();
    var fit = s.fit();
    int threads = s.learner().decodeThreads();
    var path = s.inference().kernels();
    long seed =
        CrnSeeds.derive(s.run().seed(), CrnSeeds.Domain.LEARN, span.last(), 0x76326172L, replicate);
    int count = span.last() - span.first() + 1;
    var corrections = new LedgerCorrection[count];
    var ends = new int[count];
    for (int i = 0; i < count; i++) corrections[i] = ledger(s, span.first() + i);
    try (var store = new CohortStore()) {
      for (int i = 0; i < count; i++) {
        store.addCohort(CohortWriter.directory(run, span.first() + i));
        ends[i] = store.size();
      }
      int[] order = Fit.shuffle(store.select(span.first(), span.last(), fit.rows()), seed);
      if (order.length == 0 || store.variant() != s.run().variant())
        throw new IOException("Invalid training rows");
      int steps = Math.ceilDiv(order.length, fit.batch());
      Path previous = lastCheckpoint();
      var progress =
          previous == null
              ? new Progress(0, 0, 0)
              : RecordFile.read(Progress.class, previous.resolve("progress.json")).value();
      int begin = progress.steps();
      int end = (int) Math.min(steps, (long) begin + chunkSteps);
      var rows =
          Arrays.copyOfRange(
              order,
              Math.multiplyExact(begin, fit.batch()),
              (int) Math.min(order.length, (long) end * fit.batch()));
      var plan =
          new FitPlan(
              fit.fit(steps, update.lrScale()),
              student,
              fit.auxiliary(),
              FitPlan.TOKEN_BUDGET,
              seed);
      Path stage = out.resolve("chunk.part");
      DurableFiles.delete(stage);
      Files.createDirectories(stage);
      long started = System.nanoTime();
      try (Model model =
          previous == null
              ? Fit.load(anchor, devices.primary())
              : Fit.load(previous.resolve("model"), devices.primary())) {
        var scores =
            AnchorScores.compute(store, rows, anchor, span.first(), path, devices, threads);
        var teacher =
            new Teacher(
                null,
                scores,
                1 - gain,
                (float) gain,
                id -> {
                  int index = Arrays.binarySearch(ends, id + 1);
                  return corrections[index >= 0 ? index : -1 - index];
                });
        var result =
            Fit.chunk(
                store,
                rows,
                model,
                plan,
                teacher,
                devices,
                threads,
                begin,
                previous == null ? null : previous.resolve("adam"),
                stage.resolve("adam"));
        var network = (PrismNetwork) model.getBlock();
        var after =
            new Progress(
                end,
                progress.rows() + rows.length,
                progress.elapsedSeconds() + (System.nanoTime() - started) / 1e9);
        FitResult total =
            previous == null
                ? result
                : accumulated(
                    FIT_JSON.fromJson(
                        Files.readString(previous.resolve("fit.json")), FitResult.class),
                    result,
                    devices.primary().isGpu());
        DurableFiles.write(stage.resolve("fit.json"), FIT_JSON.toJson(total));
        DurableFiles.write(stage.resolve("progress.json"), after);
        if (end == steps) {
          if (after.rows() != order.length)
            throw new IOException("The epoch did not cover every selected row");
          Snapshot.write(
              stage.resolve("snapshot"),
              network,
              PolicyParams.of(span.last() + 1),
              new Snapshot.FitRecord(
                  run.toString(), span.first(), span.last(), replicate, plan, total));
          DurableFiles.write(stage.resolve("snapshot/progress.json"), after);
          DurableFiles.seal(stage.resolve("snapshot"), out.resolve("snapshot"));
          DurableFiles.delete(stage);
        } else {
          network.save(stage.resolve("model"));
          DurableFiles.seal(
              stage,
              out.resolve("checkpoint-" + String.format(java.util.Locale.ROOT, "%09d", end)));
        }
        DurableFiles.write(out.resolve("progress.json"), after);
      }
      return Outcome.ok("Completed " + end + "/" + steps + " optimizer steps");
    }
  }

  private static FitResult accumulated(FitResult before, FitResult chunk, boolean gpu) {
    int steps = Math.addExact(before.steps(), chunk.steps());
    int count = Fit.lossSamples(steps, gpu), previous = Fit.lossSamples(before.steps(), gpu);
    double loss =
        count == previous
            ? before.loss()
            : (before.loss() * previous + chunk.loss() * (count - previous)) / count;
    double[] seconds = before.stageSeconds().clone();
    for (int i = 0; i < seconds.length; i++) seconds[i] += chunk.stageSeconds()[i];
    return new FitResult(
        Math.addExact(before.rows(), chunk.rows()),
        steps,
        before.clippedSteps() + chunk.clippedSteps(),
        loss,
        before.firstLoss(),
        count == previous ? before.lastLoss() : chunk.lastLoss(),
        before.seconds() + chunk.seconds(),
        seconds);
  }

  private Path lastCheckpoint() throws IOException {
    try (var files = Files.list(out)) {
      return files
          .filter(p -> p.getFileName().toString().matches("checkpoint-[0-9]{9}"))
          .max(Path::compareTo)
          .orElse(null);
    }
  }

  private LedgerCorrection ledger(PrismSettings s, int t) throws IOException {
    Path path = out.resolve("ledger").resolve(String.format(java.util.Locale.ROOT, "%05d.json", t));
    var luck = LuckSettlement.of(s.noise(), s.run().variant());
    if (!Files.exists(path)) {
      try (var store = new CohortStore();
          var readers = Executors.newFixedThreadPool(s.learner().decodeThreads())) {
        store.addCohort(CohortWriter.directory(run, t));
        var rows = store.select(t, t, s.fit().rows());
        var learned =
            luck.learn(
                t,
                null,
                store.iterate(rows, s.fit().batch(), readers, s.learner().decodeThreads()),
                rows.length);
        if (learned.correction() == null)
          throw new IOException("No applied ledger for cohort " + t);
        DurableFiles.write(path, learned.report().full(luck.recording()));
        return learned.correction();
      }
    }
    var saved = RecordFile.read(LedgerReport.Full.class, path).value();
    return SplitDuel.correction(saved, saved.recording(), s.run().variant());
  }
}
