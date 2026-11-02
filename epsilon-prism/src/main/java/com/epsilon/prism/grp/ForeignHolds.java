package com.epsilon.prism.grp;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 計測の窓の札: 走らせ役自身の札以外の HOLD.train* があるあいだ、裏の作業は重い計算を始めない（計測の窓を乱さない）。札は読むだけで作らない。
 */
public final class ForeignHolds {
  private static final Logger LOG = LoggerFactory.getLogger(ForeignHolds.class);

  /** 札の名前の形（prism.sh の門と同じ）。 */
  static final String PATTERN = "HOLD.train*";

  private ForeignHolds() {}

  /**
   * ほかの札がなくなるまで待つ。
   *
   * @param directory 札の置き場
   * @param ownTag 走らせ役自身の札の名前（数えない）
   * @param poll 確かめる間隔
   * @throws IOException 置き場が読めないとき
   * @throws InterruptedException 待つあいだに割り込まれたとき
   */
  public static void awaitClear(Path directory, String ownTag, Duration poll)
      throws IOException, InterruptedException {
    boolean logged = false;
    while (true) {
      List<String> holds = foreign(directory, ownTag);
      if (holds.isEmpty()) return;
      if (!logged) LOG.info("waiting for the holds {} in {}", holds, directory);
      logged = true;
      Thread.sleep(poll.toMillis());
    }
  }

  /** 走らせ役自身の札以外の札の名前（置き場がなければ空）。 */
  static List<String> foreign(Path directory, String ownTag) throws IOException {
    List<String> holds = new ArrayList<>();
    if (!Files.isDirectory(directory)) return holds;
    try (DirectoryStream<Path> listed = Files.newDirectoryStream(directory, PATTERN)) {
      for (Path hold : listed) {
        String name = hold.getFileName().toString();
        if (!name.equals(ownTag)) holds.add(name);
      }
    }
    return holds;
  }
}
