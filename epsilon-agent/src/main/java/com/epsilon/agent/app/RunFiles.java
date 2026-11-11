package com.epsilon.agent.app;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

/** run のディレクトリの snapshots/&lt;5 桁のバージョン&gt; から、公開したバージョンを探す。 */
public final class RunFiles {

  /** 公開したバージョンを並べるディレクトリ。 */
  public static final String SNAPSHOTS = "snapshots";

  private RunFiles() {}

  /** run のディレクトリの中で、指定したバージョンのディレクトリを返す。 */
  public static Path snapshot(Path root, int version) {
    return root.resolve(SNAPSHOTS).resolve(String.format("%05d", version));
  }

  /**
   * run のディレクトリの snapshots/ にあるバージョンのうち、完了マーカーのファイルまで書き終えたもので、番号がいちばん大きいものを
   * 返す。無ければ IOException を投げる。
   *
   * @param complete 書き終えたことを示すファイルの名前
   */
  public static Path latestSnapshot(Path root, String complete) throws IOException {
    Path snapshots = root.resolve(SNAPSHOTS);
    if (!Files.isDirectory(snapshots)) {
      throw new IOException("Not a run directory: " + root);
    }
    try (Stream<Path> versions = Files.list(snapshots)) {
      return versions
          .filter(path -> path.getFileName().toString().matches("\\d+"))
          .filter(path -> Files.isRegularFile(path.resolve(complete)))
          .max(Comparator.comparingInt(path -> Integer.parseInt(path.getFileName().toString())))
          .orElseThrow(() -> new IOException("No complete snapshot in " + snapshots));
    }
  }
}
