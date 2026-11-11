package com.epsilon.agent.run;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GPU の要求（~/.epsilon/gpu/requests/&lt;名前&gt;）。GPU を借りたいプロセスが置き、使い終えたら消す。中身は 1 行で、要求した
 * プロセスの pid、そのプロセスの起動時刻（{@link ProcessStart}）、実行するコマンドを空白で区切って並べる（コマンドは空白を含んでよい）。
 *
 * <p>要求が有効なのは、その pid のプロセスが同じ起動時刻のまま生きている間だけ。終了したプロセスの要求も、pid を使い回した別の
 * プロセスの要求も数えない（シェルの関数で数えるときも同じ）。段とジョブを回す JVM は、GPU の使用権を持つときだけ、セーフポイントで
 * 有効な要求に GPU を譲る。
 */
public final class GpuRequests {

  /**
   * GPU の使用権を持つことを子プロセスに伝える環境変数の名前。使用権のファイル（~/.epsilon/gpu/lease）を持つプロセス（run を動かす
   * プロセス、GPU を借りたプロセス）が設定する。
   */
  public static final String LEASE = "EPSILON_GPU_LEASE";

  private static final Logger LOG = LoggerFactory.getLogger(GpuRequests.class);

  private final Path directory;

  /** @param directory 要求を置くディレクトリ */
  public GpuRequests(Path directory) {
    this.directory = Objects.requireNonNull(directory, "directory");
  }

  /** ~/.epsilon/gpu/requests に置く要求。 */
  public static GpuRequests standard() {
    return new GpuRequests(Path.of(System.getProperty("user.home"), ".epsilon", "gpu", "requests"));
  }

  /**
   * 1 つの要求。
   *
   * @param name ファイル名
   * @param pid 要求したプロセスの pid
   * @param start そのプロセスの起動時刻
   * @param command 実行するコマンド（ログに書く）
   */
  public record Request(String name, long pid, long start, String command) {

    public Request {
      Objects.requireNonNull(name, "name");
      Objects.requireNonNull(command, "command");
    }

    /** ファイルの中身。 */
    public String line() {
      return pid + " " + start + " " + command + "\n";
    }

    /** @return 読めなければ空 */
    static Optional<Request> parse(String name, String text) {
      String[] fields = text.strip().split("\\s+", 3);
      if (fields.length < 2) {
        return Optional.empty();
      }
      String command = fields.length > 2 ? fields[2] : "";
      try {
        return Optional.of(new Request(name, Long.parseLong(fields[0]), Long.parseLong(fields[1]), command));
      } catch (NumberFormatException unreadable) {
        return Optional.empty();
      }
    }
  }

  /** 有効な要求（名前の順）。読めない要求は数えず、ログに書く。 */
  public List<Request> alive() throws IOException {
    if (!Files.isDirectory(directory)) {
      return List.of();
    }
    List<Path> files;
    try (Stream<Path> listed = Files.list(directory)) {
      files = listed.filter(path -> !path.getFileName().toString().startsWith(".")).sorted().toList();
    }
    List<Request> alive = new ArrayList<>();
    for (Path file : files) {
      String text;
      try {
        text = Files.readString(file, StandardCharsets.UTF_8);
      } catch (NoSuchFileException released) {
        continue;
      }
      Optional<Request> request = Request.parse(file.getFileName().toString(), text);
      if (request.isEmpty()) {
        LOG.warn("Ignoring the unreadable GPU request {}", file);
      } else if (ProcessStart.alive(request.get().pid(), request.get().start())) {
        alive.add(request.get());
      }
    }
    return List.copyOf(alive);
  }

  /** 要求を置く（一時ファイルに書いてから rename する）。 */
  public void write(Request request) throws IOException {
    DurableFiles.write(directory.resolve(request.name()), request.line());
  }

  /** GPU の使用権を示す環境変数があり、値が空でなければ true。 */
  public static boolean leaseHeld() {
    String lease = System.getenv(LEASE);
    return lease != null && !lease.isEmpty();
  }

  /**
   * GPU の使用権を持つことを確かめる。GPU を長く使う処理が、GPU のデバイスを開く前に呼ぶ。
   *
   * @param what 処理の名前（エラーメッセージに書く）
   * @throws IllegalStateException GPU の使用権を示す環境変数が無いか空のとき
   */
  public static void requireLease(String what) {
    if (!leaseHeld()) {
      throw new IllegalStateException(
          what + " uses the GPUs: run it under a process that holds the GPU lease and sets " + LEASE);
    }
  }
}
