package com.epsilon.reviewer.model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * チェックポイントのフィンガープリント（ファイルごとの相対パス・大きさ・更新時刻）。中身を読まずに作る。
 *
 * @param files '/' で区切った相対パス（{@code Files.walk} の結果を並べ替えた順）
 * @param sizes ファイルの大きさ（バイト）
 * @param modified 更新時刻（ミリ秒）
 */
public record CheckpointFingerprint(List<String> files, long[] sizes, long[] modified) {
  /** @throws IOException ふつうのフォルダーでないか、シンボリックリンクを含むか、ファイルが 1 つも無いとき */
  static CheckpointFingerprint of(Path directory) throws IOException {
    List<Path> paths = files(directory);
    List<String> names = new ArrayList<>(paths.size());
    long[] sizes = new long[paths.size()];
    long[] modified = new long[paths.size()];
    for (int i = 0; i < paths.size(); i++) {
      Path file = paths.get(i);
      names.add(relative(directory, file));
      sizes[i] = Files.size(file);
      modified[i] = Files.getLastModifiedTime(file).toMillis();
    }
    return new CheckpointFingerprint(List.copyOf(names), sizes, modified);
  }

  /** チェックポイントの中のふつうのファイルを、並べ替えた順で返す。 */
  static List<Path> files(Path directory) throws IOException {
    if (Files.isSymbolicLink(directory) || !Files.isDirectory(directory))
      throw new IOException("Checkpoint must be a regular directory.");
    try (var walk = Files.walk(directory)) {
      List<Path> all = walk.sorted().toList();
      if (all.stream().anyMatch(Files::isSymbolicLink))
        throw new IOException("Symbolic links are not allowed inside a checkpoint.");
      List<Path> files = all.stream().filter(Files::isRegularFile).toList();
      if (files.isEmpty()) throw new IOException("Checkpoint contains no files.");
      return files;
    }
  }

  static String relative(Path directory, Path file) {
    return directory.relativize(file).toString().replace('\\', '/');
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof CheckpointFingerprint that
        && files.equals(that.files)
        && Arrays.equals(sizes, that.sizes)
        && Arrays.equals(modified, that.modified);
  }

  @Override
  public int hashCode() {
    return 31 * (31 * files.hashCode() + Arrays.hashCode(sizes)) + Arrays.hashCode(modified);
  }
}
