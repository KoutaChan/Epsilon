package com.epsilon.agent.run;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.OptionalLong;

/**
 * プロセスの起動時刻。pid は使い回されるので、pid と起動時刻の組で 1 つのプロセスを見分ける（シェルの関数で見分けるときも同じ）。
 *
 * <p>起動時刻は /proc/&lt;pid&gt;/stat の 22 番目のフィールドで、OS の起動からの clock tick。2 番目のフィールド（プロセスの名前）は
 * 空白や括弧を含みうるので、最後の ')' の後ろを 3 番目のフィールドとして数える。終了したが親に回収されていないプロセス（状態 Z・X）は、
 * いないものとして扱う。
 */
public final class ProcessStart {

  private ProcessStart() {}

  /** @return プロセスの起動時刻。プロセスがいないか、/proc が無ければ空 */
  public static OptionalLong of(long pid) {
    String stat;
    try {
      stat = Files.readString(Path.of("/proc", Long.toString(pid), "stat"), StandardCharsets.US_ASCII);
    } catch (IOException gone) {
      return OptionalLong.empty();
    }
    return parse(stat);
  }

  /** プロセスがまだ生きていて、起動時刻も渡した値と同じなら true（pid が使い回されていれば false）。 */
  public static boolean alive(long pid, long start) {
    OptionalLong now = of(pid);
    return now.isPresent() && now.getAsLong() == start;
  }

  /** /proc/&lt;pid&gt;/stat の中身から起動時刻を読む。 */
  static OptionalLong parse(String stat) {
    int close = stat.lastIndexOf(')');
    if (close < 0) {
      return OptionalLong.empty();
    }
    // fields[0] が 3 番目のフィールド（状態）なので、22 番目は fields[19]
    String[] fields = stat.substring(close + 1).strip().split("\\s+");
    if (fields.length < 20 || fields[0].equals("Z") || fields[0].equals("X")) {
      return OptionalLong.empty();
    }
    try {
      return OptionalLong.of(Long.parseLong(fields[19]));
    } catch (NumberFormatException unreadable) {
      return OptionalLong.empty();
    }
  }
}
