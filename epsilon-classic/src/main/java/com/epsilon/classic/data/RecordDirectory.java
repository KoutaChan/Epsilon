package com.epsilon.classic.data;

import com.epsilon.mahjong.io.RecordFiles;
import com.epsilon.mahjong.record.RecordedGame;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

/**
 * 牌譜のディレクトリ。直下の通常ファイルを名前の順に並べ、1 ファイルを 1 半荘として読む。ファイルの形式は {@link RecordFiles} が
 * 見分ける。
 */
public final class RecordDirectory {

  private RecordDirectory() {}

  /** 直下の通常ファイルを名前の順に返す。名前が . で始まるファイルと、一時ファイル（*.tmp）は除く。 */
  public static List<Path> files(Path directory) throws IOException {
    try (Stream<Path> listed = Files.list(directory)) {
      return listed
          .filter(Files::isRegularFile)
          .filter(file -> !file.getFileName().toString().startsWith("."))
          .filter(file -> !file.getFileName().toString().endsWith(".tmp"))
          .sorted()
          .toList();
    }
  }

  /** ファイルを 1 半荘として読む。牌譜 id は、ファイル名から最初の . 以降を除いたもの。 */
  public static RecordedGame read(Path file) throws IOException {
    return RecordFiles.read(Files.readAllBytes(file), id(file));
  }

  /** ファイル名から求めた牌譜 id。 */
  public static String id(Path file) {
    String name = file.getFileName().toString();
    int dot = name.indexOf('.');
    return dot > 0 ? name.substring(0, dot) : name;
  }
}
