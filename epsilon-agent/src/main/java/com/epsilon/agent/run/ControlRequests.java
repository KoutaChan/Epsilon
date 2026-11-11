package com.epsilon.agent.run;

import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * run の control に置く要求。置いた要求は書き換えない。段とジョブを回す JVM は、従った要求と取り消した要求を control/done に移すだけに
 * する。要求のファイルは一時ファイルから rename で置くので、読む側に書きかけは見えない。
 *
 * <p>止めるときは、停止マーカー（control/stopped）を置いてから、停止の要求を移す。
 */
public final class ControlRequests {

  private static final Pattern NAME = Pattern.compile("(\\d+)-([a-z]+(?:-[a-z]+)*)\\.json");

  /** 同じ JVM の中で、番号を決めるためのロックを取るスレッドを 1 つずつにする。 */
  private static final Object NUMBERING = new Object();

  private final RunDirectory run;

  public ControlRequests(RunDirectory run) {
    this.run = run;
  }

  /**
   * 要求を置く。番号は、control と control/done にある番号の最大値 + 1。番号を決めてから置くまでを、control/.lock のロックを
   * 取ったまま行う。
   *
   * @return 置いた要求
   */
  public ControlRequest submit(Request request) throws IOException {
    Path control = run.control();
    DurableFiles.createDirectories(control);
    synchronized (NUMBERING) {
      try (FileChannel channel =
              FileChannel.open(control.resolve(".lock"), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
          FileLock lock = channel.lock()) {
        // JVM は control から done へ移すので、control を先に見れば、移している途中の要求も必ずどちらかで数えられる
        int highest = highest(control);
        highest = Math.max(highest, highest(done()));
        ControlRequest written = new ControlRequest(highest + 1, Instant.now(), request);
        DurableFiles.write(control.resolve(written.fileName()), written);
        return written;
      }
    }
  }

  /**
   * まだ従っていない要求（番号の順）。
   *
   * @throws SettingsException 読めない要求のファイルがあるか、ファイル名と中身が合わない
   */
  public List<ControlRequest> pending() throws IOException {
    return read(run.control());
  }

  /**
   * 従ったか取り消した要求を、control/done に移す。セーフポイントの確かめと GRP の裏の作業は別のスレッドで要求を移すので、もう移して
   * あれば何もしない。
   */
  public void retire(ControlRequest request) throws IOException {
    Path done = done();
    DurableFiles.createDirectories(done);
    Path retired = done.resolve(request.fileName());
    try {
      Files.move(run.control().resolve(request.fileName()), retired, StandardCopyOption.ATOMIC_MOVE);
    } catch (NoSuchFileException moved) {
      if (Files.exists(retired)) {
        return;
      }
      throw moved;
    }
    DurableFiles.syncDirectory(done);
    DurableFiles.syncDirectory(run.control());
  }

  /**
   * 停止マーカーを置く。停止の要求を control/done に移す前に呼ぶ。
   *
   * @param reason 止まった理由（英語の 1 行）
   */
  public void markStopped(String reason) throws IOException {
    DurableFiles.write(stoppedMark(), reason + "\n");
  }

  /** 停止マーカーがあれば true。 */
  public boolean stopped() {
    return Files.exists(stoppedMark());
  }

  /** 停止マーカーに書いた理由。マーカーが無ければ空の文字列。 */
  public String stoppedReason() throws IOException {
    Path mark = stoppedMark();
    return Files.exists(mark) ? Files.readString(mark, StandardCharsets.UTF_8).strip() : "";
  }

  private Path stoppedMark() {
    return run.control().resolve("stopped");
  }

  private Path done() {
    return run.control().resolve("done");
  }

  private static List<ControlRequest> read(Path directory) throws IOException {
    if (!Files.isDirectory(directory)) {
      return List.of();
    }
    List<Path> files;
    try (Stream<Path> listed = Files.list(directory)) {
      files = listed.filter(path -> NAME.matcher(path.getFileName().toString()).matches()).toList();
    }
    List<ControlRequest> requests = new ArrayList<>(files.size());
    for (Path file : files) {
      ControlRequest request;
      try {
        request = RecordFile.read(ControlRequest.class, file).value();
      } catch (NoSuchFileException retired) {
        // 一覧を作った後に control/done へ移った要求
        continue;
      }
      if (!request.fileName().equals(file.getFileName().toString())) {
        throw new SettingsException(
            List.of("The request file " + file + " holds request " + request.fileName() + " instead"));
      }
      requests.add(request);
    }
    requests.sort(Comparator.comparingInt(ControlRequest::number));
    return List.copyOf(requests);
  }

  private static int highest(Path directory) throws IOException {
    if (!Files.isDirectory(directory)) {
      return 0;
    }
    try (Stream<Path> listed = Files.list(directory)) {
      return listed
          .map(path -> NAME.matcher(path.getFileName().toString()))
          .filter(Matcher::matches)
          .mapToInt(matcher -> Integer.parseInt(matcher.group(1)))
          .max()
          .orElse(0);
    }
  }
}
