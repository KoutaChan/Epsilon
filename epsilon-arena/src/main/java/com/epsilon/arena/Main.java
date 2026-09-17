package com.epsilon.arena;

import com.epsilon.major.EpsilonModelProvider;
import com.epsilon.nano.NanoModelProvider;
import com.epsilon.pico.PicoModelProvider;
import com.epsilon.spi.PolicyExecutionContext;
import com.google.gson.GsonBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 系列ごとの実行 jar を読み込み直さず、同じ JVM で混合対戦を起動する。 */
public final class Main {
  private static final Logger log = LoggerFactory.getLogger(Main.class);

  private Main() {}

  public static void main(String[] args) throws Exception {
    if (args.length == 0 || args[0].equals("--help") || args[0].equals("help")) {
      ArenaConfig.defaultPath();
      log.info(
          """
          epsilon-arena duel --candidate-series epsilon-nano --candidate-checkpoint PATH
                             --opponent-series epsilon --opponent-checkpoint PATH [options]
          epsilon-arena match [--config PATH] [--warmup-games 32 --repetitions 3 --output report.json]

          duel options: --games 256 --concurrent-games 128 --workers N --seed 1
                        --max-batch-wait-micros 1000000 --first-wall-family 0 --device cpu|gpu:0|auto
                        --candidate-settings PATH --opponent-settings PATH
                        --slots-per-device 2 --ready-batches-per-device 4
          games / warmup-games must be multiples of 4. Each wall seed is evaluated with all four seat rotations.
          Default match settings: config/arena.toml (created on first use; existing files are preserved)\
          """);
      return;
    }
    String command = args[0];
    if (!command.equals("duel") && !command.equals("match"))
      throw new IllegalArgumentException("Expected duel or match: " + command);
    Set<String> allowed =
        command.equals("match")
            ? Set.of("config", "warmup-games", "repetitions", "output")
            : Set.of(
                "candidate-series",
                "candidate-checkpoint",
                "opponent-series",
                "opponent-checkpoint",
                "candidate-settings",
                "opponent-settings",
                "device",
                "games",
                "concurrent-games",
                "workers",
                "seed",
                "first-wall-family",
                "max-batch-wait-micros",
                "slots-per-device",
                "ready-batches-per-device",
                "warmup-games",
                "repetitions",
                "output");
    Map<String, String> options = parseOptions(args, allowed);
    ArenaConfig config =
        command.equals("match")
            ? ArenaConfig.load(
                options.containsKey("config")
                    ? Path.of(required(options, "config"))
                    : ArenaConfig.defaultPath())
            : duel(options);
    int repetitions = Integer.parseInt(options.getOrDefault("repetitions", "1"));
    int warmupGames = Integer.parseInt(options.getOrDefault("warmup-games", "0"));
    if (repetitions < 1 || warmupGames < 0 || warmupGames % 4 != 0)
      throw new IllegalArgumentException(
          "repetitions must be positive; warmup-games must be a nonnegative multiple of 4");

    try (PolicyExecutionContext execution =
            new PolicyExecutionContext(
                config.run().workers(),
                config.run().slotsPerDevice(),
                config.run().readyBatchesPerDevice());
        ModelRegistry registry =
            new ModelRegistry(
                List.of(
                    new NanoModelProvider(), new PicoModelProvider(), new EpsilonModelProvider()),
                execution)) {
      List<Participant> participants = new ArrayList<>(4);
      for (ModelSpec spec : config.seats()) participants.add(registry.open(spec));
      if (warmupGames != 0) {
        RunSettings run = config.run();
        new ArenaRunner(
                participants,
                new RunSettings(
                    warmupGames,
                    run.concurrentGames(),
                    run.workers(),
                    run.seed(),
                    run.maxBatchWaitMicros(),
                    run.firstWallFamily(),
                    run.slotsPerDevice(),
                    run.readyBatchesPerDevice()))
            .run();
      }
      List<ArenaResult> runs = new ArrayList<>(repetitions);
      List<Map<String, Object>> gpuMemory = new ArrayList<>(repetitions);
      RunSettings run = config.run();
      RunSettings measured =
          new RunSettings(
              run.games(),
              run.concurrentGames(),
              run.workers(),
              run.seed(),
              run.maxBatchWaitMicros(),
              Math.addExact(run.firstWallFamily(), warmupGames / 4),
              run.slotsPerDevice(),
              run.readyBatchesPerDevice());
      for (int index = 0; index < repetitions; index++) {
        var before = execution.memorySnapshot();
        runs.add(new ArenaRunner(participants, measured).run());
        gpuMemory.add(
            Map.of("repetition", index + 1, "before", before, "after", execution.memorySnapshot()));
      }
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("command", command);
      report.put("java", System.getProperty("java.runtime.version"));
      report.put("os", System.getProperty("os.name"));
      report.put(
          "models",
          config.seats().stream()
              .map(
                  spec ->
                      Map.of(
                          "series",
                          spec.series(),
                          "checkpoint",
                          spec.checkpoint().toAbsolutePath().normalize().toString(),
                          "options",
                          spec.options()))
              .toList());
      report.put("warmupGames", warmupGames);
      report.put("gpuMemory", gpuMemory);
      report.put("runs", runs);
      String json = new GsonBuilder().setPrettyPrinting().create().toJson(report);
      if (options.containsKey("output")) {
        Path output = Path.of(options.get("output")).toAbsolutePath();
        Files.createDirectories(output.getParent());
        Files.writeString(output, json + System.lineSeparator());
      }
      log.info("\n{}", json);
    }
  }

  private static ArenaConfig duel(Map<String, String> options) {
    Map<String, String> overrides = new LinkedHashMap<>();
    Map.of(
            "games", "arena.games",
            "concurrent-games", "arena.concurrentGames",
            "workers", "arena.workers",
            "seed", "arena.seed",
            "max-batch-wait-micros", "arena.maxBatchWaitMicros",
            "first-wall-family", "arena.firstWallFamily",
            "slots-per-device", "arena.slotsPerDevice",
            "ready-batches-per-device", "arena.readyBatchesPerDevice")
        .forEach(
            (option, setting) -> {
              if (options.containsKey(option)) overrides.put(setting, options.get(option));
            });
    RunSettings run =
        ArenaConfig.defaultSettings().withOverrides(overrides).bind(RunSettings.class);
    ModelSpec candidate = spec("candidate", options);
    ModelSpec opponent = spec("opponent", options);
    return new ArenaConfig(run, List.of(candidate, opponent, opponent, opponent));
  }

  private static ModelSpec spec(String side, Map<String, String> options) {
    Map<String, String> modelOptions = new LinkedHashMap<>();
    if (options.containsKey("device")) modelOptions.put("device", options.get("device"));
    if (options.containsKey(side + "-settings"))
      modelOptions.put("settings", options.get(side + "-settings"));
    return new ModelSpec(
        required(options, side + "-series"),
        Path.of(required(options, side + "-checkpoint")),
        modelOptions);
  }

  private static Map<String, String> parseOptions(String[] args, Set<String> allowed) {
    Map<String, String> result = new LinkedHashMap<>();
    for (int index = 1; index < args.length; index += 2) {
      String argument = args[index];
      if (!argument.startsWith("--")
          || !allowed.contains(argument.substring(2))
          || index + 1 == args.length)
        throw new IllegalArgumentException("Unknown option or missing value: " + argument);
      String key = argument.substring(2);
      if (result.put(key, args[index + 1]) != null)
        throw new IllegalArgumentException("Repeated option: " + argument);
    }
    return result;
  }

  private static String required(Map<String, String> options, String name) {
    String value = options.get(name);
    if (value == null || value.isBlank()) throw new IllegalArgumentException("Missing --" + name);
    return value;
  }
}
