package com.epsilon.arena;

import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsFiles;
import com.epsilon.config.settings.SettingsLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 型付きの実行設定と四つの参加枠を読み込む。相対パスは選択した設定ファイル基準。 */
public record ArenaConfig(RunSettings run, List<ModelSpec> seats) {
  public static final String DEFAULT_RESOURCE = "arena/settings.toml";
  private static final Path DEFAULT_PATH = Path.of("config", "arena.toml");
  private static final Set<String> ARENA_KEYS =
      Set.of(
          "arena.games",
          "arena.concurrentGames",
          "arena.workers",
          "arena.seed",
          "arena.maxBatchWaitMicros",
          "arena.firstWallFamily",
          "arena.slotsPerDevice",
          "arena.readyBatchesPerDevice");

  /** 同梱設定を初回だけ生成し、通常の対局で使う設定パスを返す。 */
  public static Path defaultPath() throws IOException {
    SettingsFiles.createDefault(DEFAULT_PATH, DEFAULT_RESOURCE);
    return DEFAULT_PATH;
  }

  /** 同梱する実行既定値へ選択ファイルを重ねる。参加枠は選択ファイルで必ず指定する。 */
  public static ArenaConfig load(Path path) throws IOException {
    SettingsLoader settings = SettingsLoader.load(defaultSettings(), path, Map.of());
    Map<String, String> values = settings.values();
    for (String key : values.keySet()) {
      if (!ARENA_KEYS.contains(key) && !key.matches("seat[0-3]\\.(series|checkpoint|options\\..+)"))
        throw new IllegalArgumentException("Unknown arena setting: " + key);
    }
    RunSettings run = settings.bind(RunSettings.class);
    Path base = path.toAbsolutePath().getParent();
    List<ModelSpec> seats = new ArrayList<>(4);
    for (int seat = 0; seat < 4; seat++) {
      String prefix = "seat" + seat;
      SeatSettings participant = settings.bind(SeatSettings.class, prefix);
      String optionsPrefix = prefix + ".options.";
      Map<String, String> options = new LinkedHashMap<>();
      values.forEach(
          (key, value) -> {
            if (key.startsWith(optionsPrefix))
              options.put(key.substring(optionsPrefix.length()), value);
          });
      if (options.containsKey("settings"))
        options.put("settings", base.resolve(options.get("settings")).normalize().toString());
      seats.add(
          new ModelSpec(
              participant.series(), base.resolve(participant.checkpoint()).normalize(), options));
    }
    return new ArenaConfig(run, List.copyOf(seats));
  }

  /** 対局と比較対局が共有する実行既定値。参加枠の配置例は既定値として補完しない。 */
  static SettingsLoader defaultSettings() {
    return Defaults.VALUE;
  }

  private record SeatSettings(
      @Setting("series") String series, @Setting("checkpoint") String checkpoint) {}

  private static final class Defaults {
    private static final SettingsLoader VALUE = read();

    private static SettingsLoader read() {
      Map<String, String> values = new LinkedHashMap<>();
      SettingsLoader.fromResource(DEFAULT_RESOURCE)
          .values()
          .forEach(
              (key, value) -> {
                if (key.startsWith("arena.")) values.put(key, value);
              });
      // ワーカー数だけは実行環境で決める。選択ファイルやCLIの明示値を優先する。
      values.putIfAbsent(
          "arena.workers",
          Integer.toString(Math.max(1, Runtime.getRuntime().availableProcessors() / 2)));
      return SettingsLoader.of(values);
    }
  }
}
