package com.epsilon.prism.run;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

/**
 * バージョン V の x1n3 の出力（x1n3Dir の {@code v<V>-x1n3.*}）。印のファイルが測りの状態を表す:
 *
 * <ul>
 *   <li>{@code .running}: 測りの最中（網を開く前に作り、{@code .done} の後に消す）。
 *   <li>{@code .tsv}・{@code .log}: 半荘ごとの結果（{@link NanoTable#perGame}）と集計の行。どちらも {@code .tmp} に書いてから差し替える。
 *   <li>{@code .done}: 書き終えた印（これがあれば測らない）。
 *   <li>{@code .failed}: 測れなかったバージョン（これがあれば測らない。測り直すときは人が消す）。
 * </ul>
 *
 * 途中で止まると {@code .done} がないので、次の起動で初めから測り直す（{@code .tmp} は上書き）。
 *
 * @param directory 出力のディレクトリ
 * @param version バージョン
 */
public record X1n3Outputs(Path directory, int version) {

  /** 測りの最中の印。 */
  public Path running() {
    return file("running");
  }

  /** 書き終えた印。 */
  public Path done() {
    return file("done");
  }

  /** 測れなかったバージョンの印。 */
  public Path failed() {
    return file("failed");
  }

  /** 半荘ごとの結果。 */
  public Path tsv() {
    return file("tsv");
  }

  /** 集計の行。 */
  public Path log() {
    return file("log");
  }

  /** {@code .done} か {@code .failed} があれば true（測らない）。 */
  public boolean settled() {
    return Files.exists(done()) || Files.exists(failed());
  }

  /** 測りを始める: {@code .running} を作る（中身は JVM の pid）。 */
  public void begin() throws IOException {
    Files.createDirectories(directory);
    Files.writeString(running(), ProcessHandle.current().pid() + "\n", StandardCharsets.US_ASCII);
  }

  /** 結果を書く: tsv と log を差し替え、{@code .done} を作ってから {@code .running} を消す。 */
  public void commit(String perGame, String log) throws IOException {
    RunFiles.writeAtomically(tsv(), perGame);
    RunFiles.writeAtomically(log(), log);
    Files.writeString(done(), "", StandardCharsets.US_ASCII);
    Files.deleteIfExists(running());
  }

  /** 測れなかった: log を差し替え、{@code .running} を {@code .failed} にする。 */
  public void fail(String log) throws IOException {
    RunFiles.writeAtomically(log(), log);
    if (Files.exists(running()))
      Files.move(running(), failed(), StandardCopyOption.REPLACE_EXISTING);
    else Files.writeString(failed(), "", StandardCharsets.US_ASCII);
  }

  /**
   * 測っていないバージョン: x1n3From ≤ V ≤ latest で V が x1n3Every の倍数、{@code .done} も {@code .failed} もないバージョン（古い順）。x1n3 = OFF なら空。
   *
   * @param latest 公開した最新のバージョン
   */
  public static List<Integer> pending(PrismConfig config, int latest) {
    List<Integer> versions = new ArrayList<>();
    if (config.x1n3() == X1n3.OFF) return versions;
    Path directory = Path.of(config.x1n3Dir());
    int every = config.x1n3Every();
    int first = Math.ceilDiv(config.x1n3From(), every) * every;
    for (int version = first; version <= latest; version += every)
      if (!new X1n3Outputs(directory, version).settled()) versions.add(version);
    return versions;
  }

  private Path file(String suffix) {
    return directory.resolve("v" + version + "-x1n3." + suffix);
  }
}
