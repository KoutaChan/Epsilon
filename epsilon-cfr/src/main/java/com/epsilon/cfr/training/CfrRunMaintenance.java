package com.epsilon.cfr.training;

import com.epsilon.cfr.data.CfrReplayStore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

/**
 * 保存が確定した工程を基準に、現在の学習区間で不要になった対局の一時ファイルと再生データの世代を削除する。
 * 古い状態JSONだけを使った手動復元は保証しない。モデル、他の学習区間のデータは削除しない。
 */
public final class CfrRunMaintenance {
  private CfrRunMaintenance() {}

  public static void prune(CfrRunStore store, CfrRunState state) throws IOException {
    if (!store.read().equals(state))
      throw new IOException("Cannot prune from an unpublished CFR state");
    switch (state.phase()) {
      case READY, COLLECTING, SEALED -> {
        return;
      }
      default -> {}
    }
    if (state.replayGeneration() <= 0)
      throw new IOException("Committed phase has no replay generation");
    Path solve = store.resolve("solves/" + state.solveId());
    if (!solve.getParent().equals(store.resolve("solves"))
        || !Files.isDirectory(solve, LinkOption.NOFOLLOW_LINKS))
      throw new IOException("Invalid CFR solve directory");
    Path replayDirectory = solve.resolve("replay");
    if (!Files.isDirectory(replayDirectory, LinkOption.NOFOLLOW_LINKS))
      throw new IOException("Invalid CFR replay directory");
    // 実行状態が指す世代を読み込み、削除前に保持中の参照先をすべて検証する。
    try (CfrReplayStore replay =
        CfrReplayStore.loadGeneration(replayDirectory, state.solveId(), state.replayGeneration())) {
      TreeMap<Long, Path> snapshots = snapshots(replayDirectory);
      Long previous = snapshots.lowerKey(state.replayGeneration());
      // 異常終了により、replay/current.binだけが実行状態より先の世代を指している場合がある。
      // 先に参照先を戻し、削除の途中で停止しても両方の参照先を読み込めるようにする。
      replay.restoreCurrentPointer();
      for (Map.Entry<Long, Path> snapshot : snapshots.entrySet()) {
        if (snapshot.getKey() != state.replayGeneration() && !snapshot.getKey().equals(previous))
          Files.delete(snapshot.getValue());
      }
      replay.collectGarbage();
    }
    Path matches = solve.resolve("matches");
    if (!Files.isDirectory(matches, LinkOption.NOFOLLOW_LINKS)) return;
    try (var directories = Files.list(matches)) {
      for (Path iteration :
          directories.filter(path -> Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)).toList()) {
        String name = iteration.getFileName().toString();
        if (!name.matches("[0-9]+")) continue;
        long number;
        try {
          number = Long.parseLong(name);
        } catch (NumberFormatException invalid) {
          continue;
        }
        if (number > state.iteration()) continue;
        // 削除対象がこの学習区間の一時保存ディレクトリ内にあることを確認してから、再帰的に削除する。
        // Files.walkはシンボリックリンクをたどらない。
        Path target = iteration.toAbsolutePath().normalize();
        if (!target.startsWith(matches.toAbsolutePath().normalize()))
          throw new IOException("Match spool escaped its solve directory");
        try (var files = Files.walk(target)) {
          for (Path path : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
        }
      }
    }
  }

  private static TreeMap<Long, Path> snapshots(Path directory) throws IOException {
    TreeMap<Long, Path> result = new TreeMap<>();
    try (var files = Files.list(directory)) {
      for (Path file :
          files.filter(path -> Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)).toList()) {
        String name = file.getFileName().toString();
        if (!name.matches("replay-[0-9]+\\.bin")) continue;
        try {
          result.put(Long.parseLong(name.substring(7, name.length() - 4)), file);
        } catch (NumberFormatException invalid) {
          throw new IOException("Invalid replay generation: " + name, invalid);
        }
      }
    }
    return result;
  }
}
