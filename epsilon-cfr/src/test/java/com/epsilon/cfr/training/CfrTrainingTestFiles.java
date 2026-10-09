package com.epsilon.cfr.training;

import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrEpisodeFrame;
import com.epsilon.cfr.data.CfrEpisodeWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** 保存・再開の検証用に、指定された経路を一つの完了済み半荘として保存する。 */
final class CfrTrainingTestFiles {
  private CfrTrainingTestFiles() {}

  static void writeMatch(Path path, CfrEpisode... trajectories) throws IOException {
    if (trajectories.length == 0) throw new IllegalArgumentException("Missing test trajectories");
    try (var writer = new CfrEpisodeWriter(path.getParent())) {
      for (int index = 0; index < trajectories.length; index++) {
        writer.accept(
            new CfrEpisodeFrame(trajectories[index], index, index + 1 == trajectories.length));
      }
    }
    Path actual =
        path.resolveSibling(String.format(Locale.ROOT, "%08d.bin", trajectories[0].episodeId()));
    if (!actual.equals(path)) Files.move(actual, path);
  }
}
