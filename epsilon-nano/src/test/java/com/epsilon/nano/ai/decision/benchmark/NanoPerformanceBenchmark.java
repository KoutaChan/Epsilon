package com.epsilon.nano.ai.decision.benchmark;

import com.epsilon.nano.ai.decision.training.NanoProductionBenchmarkSupport;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.google.gson.GsonBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * 本番自己対局で保存した教師を、本番の逐次読み込み learner で更新する明示起動専用測定。
 *
 * <p>SETTINGS CHECKPOINT_ROOT OUTPUT [GAMES=512] [WARMUP_PASSES=1] [MEASURED_PASSES=1]
 * [SEED=20261008]。SETTINGS=- は同梱設定、OUTPUT は未作成のディレクトリを指定する。 同じ収集済みデータを再利用し、元チェックポイントは更新しない。
 */
public final class NanoPerformanceBenchmark {
  private NanoPerformanceBenchmark() {}

  public static void main(String[] args) throws Exception {
    if (args.length < 3 || args.length > 7)
      throw new IllegalArgumentException(
          "SETTINGS CHECKPOINT_ROOT OUTPUT [GAMES=512] [WARMUP_PASSES=1]"
              + " [MEASURED_PASSES=1] [SEED=20261008]");
    int games = positive(args, 3, 512);
    int warmupPasses = positive(args, 4, 1);
    int measuredPasses = positive(args, 5, 1);
    long seed = args.length > 6 ? Long.parseLong(args[6]) : 20261008L;
    var settings =
        args[0].equals("-")
            ? EpsilonSettings.defaults()
            : EpsilonSettings.load(Path.of(args[0]), Map.of());
    Path checkpoint = Path.of(args[1]).toAbsolutePath().normalize();
    Path output = Path.of(args[2]).toAbsolutePath().normalize();
    Files.createDirectories(output.getParent());
    Files.createDirectory(output);
    var json = new GsonBuilder().setPrettyPrinting().create();
    Files.writeString(output.resolve("effective-settings.json"), json.toJson(settings.values()));
    Map<String, Object> report =
        NanoProductionBenchmarkSupport.run(
            checkpoint,
            output.resolve("dataset"),
            settings,
            games,
            warmupPasses,
            measuredPasses,
            seed);
    String result = json.toJson(report);
    Files.writeString(output.resolve("report.json"), result + System.lineSeparator());
    System.out.println(result);
  }

  private static int positive(String[] args, int index, int fallback) {
    int value = args.length > index ? Integer.parseInt(args[index]) : fallback;
    if (value < 1) throw new IllegalArgumentException("Argument " + index + " must be positive");
    return value;
  }
}
