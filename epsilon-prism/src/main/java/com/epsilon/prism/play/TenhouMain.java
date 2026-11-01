package com.epsilon.prism.play;

import com.epsilon.client.tenhou.TenhouClient;
import com.epsilon.prism.config.settings.EpsilonSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 公開した版（snapshots/&lt;t&gt;）で天鳳の個室を打つ。手は版の s の貪欲（{@link GreedyPlayer}）。
 *
 * <p>引数は共通の天鳳クライアント（{@link TenhouClient.Options}。{@code --checkpoint} は版のディレクトリ）に、{@code --settings &lt;toml&gt;}
 * （系列の設定。既定は同梱）と {@code --device &lt;cpu|gpu:0&gt;}（epsilon.devices.inference。既定 cpu、CPU は FP32 の eager）を足す。
 */
public final class TenhouMain {
  private static final Logger log = LoggerFactory.getLogger(TenhouMain.class);

  private TenhouMain() {}

  public static void main(String[] args) throws IOException {
    String settings = null;
    String device = "cpu";
    List<String> rest = new ArrayList<>();
    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--settings" -> settings = value(args, ++i, "--settings");
        case "--device" -> device = value(args, ++i, "--device");
        default -> rest.add(args[i]);
      }
    }
    var options = TenhouClient.Options.parse(rest.toArray(String[]::new), "snapshots/latest");
    if (options == null) {
      TenhouClient.printUsage("snapshots/latest");
      System.exit(1);
      return;
    }
    if (options.samplePolicy()) throw new IllegalArgumentException("PRISM plays the greedy policy; --sample is not supported");
    Map<String, String> overrides = new LinkedHashMap<>();
    overrides.put("epsilon.devices.inference", device);
    try (SnapshotScorer scorer =
        SnapshotScorer.open(
            Path.of(options.checkpointDir()),
            EpsilonSettings.load(settings == null ? null : Path.of(settings), overrides))) {
      log.info("PRISM snapshot {} on {}", options.checkpointDir(), device);
      TenhouClient.run(new GreedyPlayer(scorer), options);
    } catch (Exception failure) {
      log.error("Fatal error", failure);
      System.exit(1);
    }
  }

  private static String value(String[] args, int index, String flag) {
    if (index >= args.length) throw new IllegalArgumentException(flag + " needs a value");
    return args[index];
  }
}
