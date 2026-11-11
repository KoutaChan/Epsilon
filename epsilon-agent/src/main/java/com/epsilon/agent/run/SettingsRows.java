package com.epsilon.agent.run;

import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * run の設定の行（settings.jsonl）。行を足すことと、反復に行を固定することは、同じロック（settings.lock）を取って行う。
 *
 * <p>行を足すときは、ファイルを丸ごと一時ファイルに書いてから rename するので、書いている途中で落ちても前のファイルが残る。壊れた行は
 * エラーにして、読み飛ばさない。
 */
public final class SettingsRows {

  /** 同じ JVM の中では、ロックを 1 つずつ取らせる（POSIX のロックはプロセスごとなので、同じ JVM が 2 つ目を取ろうとすると例外になる）。 */
  private static final ReentrantLock IN_JVM = new ReentrantLock();

  private final RunDirectory run;

  public SettingsRows(RunDirectory run) {
    this.run = run;
  }

  /**
   * 全ての行（番号の順）。ファイルが無ければ空。
   *
   * @throws SettingsException 読めない行があるか、ファイルが改行で終わっていないか、番号が 1 から 1 つずつ増えていない
   */
  public List<SettingsRow> read() throws IOException {
    Path file = run.settings();
    if (!Files.exists(file)) {
      return List.of();
    }
    String text = Files.readString(file, StandardCharsets.UTF_8);
    if (text.isEmpty()) {
      return List.of();
    }
    if (!text.endsWith("\n")) {
      throw new SettingsException(List.of(file + " does not end with a newline"));
    }
    String[] lines = text.substring(0, text.length() - 1).split("\n", -1);
    List<SettingsRow> rows = new ArrayList<>(lines.length);
    for (int index = 0; index < lines.length; index++) {
      String origin = file + ":" + (index + 1);
      SettingsRow row = RecordFile.read(SettingsRow.class, lines[index], origin).value();
      if (row.row() != index + 1) {
        throw new SettingsException(
            List.of("Settings row " + row.row() + " at " + origin + " is out of order (expected " + (index + 1) + ")"));
      }
      rows.add(row);
    }
    return List.copyOf(rows);
  }

  /**
   * 最新の行。
   *
   * @throws SettingsException 行が 1 つも無い
   */
  public SettingsRow latest() throws IOException {
    List<SettingsRow> rows = read();
    if (rows.isEmpty()) {
      throw new SettingsException(List.of("No settings rows in " + run.settings() + ": create the run first"));
    }
    return rows.getLast();
  }

  /**
   * 指定した番号の行。
   *
   * @throws SettingsException その番号の行が無い
   */
  public SettingsRow row(int number) throws IOException {
    List<SettingsRow> rows = read();
    if (number < 1 || number > rows.size()) {
      throw new SettingsException(List.of("No settings row " + number + " in " + run.settings()));
    }
    return rows.get(number - 1);
  }

  /**
   * ロックを取る。ほかのプロセスや、この JVM のほかのスレッドが持っていれば待つ。
   *
   * @throws IllegalStateException このスレッドが既にロックを持っている
   */
  public Locked lock() throws IOException {
    if (IN_JVM.isHeldByCurrentThread()) {
      throw new IllegalStateException("This thread already holds a settings lock");
    }
    IN_JVM.lock();
    try {
      DurableFiles.createDirectories(run.root());
      FileChannel channel = FileChannel.open(run.settingsLock(), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
      try {
        channel.lock();
      } catch (IOException | RuntimeException failure) {
        channel.close();
        throw failure;
      }
      return new Locked(channel);
    } catch (IOException | RuntimeException failure) {
      IN_JVM.unlock();
      throw failure;
    }
  }

  /**
   * state.json にある次の反復の番号（state.json が無ければ 1）。どの run の種類の state も、次の反復を {@code next} のフィールドに持つ
   * （{@link Progress}）。
   *
   * @throws SettingsException state.json が読めないか、{@code next} が整数でない
   */
  static int nextIteration(RunDirectory run) throws IOException {
    Path state = run.state();
    if (!Files.exists(state)) {
      return 1;
    }
    Node next = Json.read(Files.readString(state, StandardCharsets.UTF_8), state.toString()).get("next");
    if (next instanceof Node.Numeral numeral && numeral.integer()) {
      return numeral.decimal().intValueExact();
    }
    throw new SettingsException(List.of("Field next of " + state + " must be an integer"));
  }

  /** settings.lock を持っている間に行う操作。閉じるとロックを外す。 */
  public final class Locked implements AutoCloseable {

    private final FileChannel channel;

    private Locked(FileChannel channel) {
      this.channel = channel;
    }

    public List<SettingsRow> rows() throws IOException {
      return read();
    }

    public SettingsRow latest() throws IOException {
      return SettingsRows.this.latest();
    }

    public SettingsRow row(int number) throws IOException {
      return SettingsRows.this.row(number);
    }

    /**
     * 行を 1 つ足す。番号は最新の行の次、時刻は今の時刻にする。前の行は 1 文字も変えない。
     *
     * @param from 効き始める反復
     * @param jar 行を書く jar の ID
     * @param reason 変えた理由
     * @param changes 前の行から変えたキー
     * @param settings 全てのキーの値
     * @return 足した行
     */
    public SettingsRow append(
        int from, String jar, String reason, List<SettingsRow.ChangedKey> changes, Node.Table settings)
        throws IOException {
      List<SettingsRow> rows = read();
      SettingsRow row = new SettingsRow(rows.size() + 1, from, Instant.now(), jar, reason, changes, settings);
      String before = Files.exists(run.settings()) ? Files.readString(run.settings(), StandardCharsets.UTF_8) : "";
      DurableFiles.write(run.settings(), before + RecordFile.json(row) + "\n");
      return row;
    }

    /** 今足す行が効き始める反復。ふつうは state の次の反復で、その反復をもう固定していれば、その次の反復。 */
    public int firstOpenIteration() throws IOException {
      int next = nextIteration(run);
      return Files.exists(run.iterationFile(next)) ? next + 1 : next;
    }

    @Override
    public void close() throws IOException {
      try {
        channel.close();
      } finally {
        IN_JVM.unlock();
      }
    }
  }
}
