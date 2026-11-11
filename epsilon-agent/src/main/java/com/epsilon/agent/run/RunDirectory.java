package com.epsilon.agent.run;

import java.nio.file.Path;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * run のディレクトリ。中のファイルの名前と置き方は、ここにだけ書く。反復とバージョンの番号は 0 埋めの 5 桁。
 *
 * <pre>
 * settings.jsonl  settings.lock  state.json  status.json
 * control/（要求、done/、stopped）  .driver.lock  .train.lock
 * work/&lt;反復&gt;/ITERATION.json  rows/&lt;反復&gt;/  snapshots/&lt;バージョン&gt;/  grp/
 * jobs/&lt;ジョブ&gt;/  champion/  reports/  logs/
 * </pre>
 *
 * @param root run のディレクトリ（絶対パスに直す）
 */
public record RunDirectory(Path root) {

  /** 反復の報告を書くファイルの名前（reports の中）。 */
  public static final String ITERATIONS = "iterations.jsonl";

  /** 終わったジョブの結果を書くファイルの名前（reports の中）。 */
  public static final String EVALUATIONS = "evaluations.jsonl";

  private static final Pattern JOB = Pattern.compile("v(\\d+)-(.+)");

  public RunDirectory {
    root = root.toAbsolutePath().normalize();
  }

  /** 設定の行を並べたファイル。 */
  public Path settings() {
    return root.resolve("settings.jsonl");
  }

  /** 設定の行を足すときと、反復に行を固定するときに取るロック。 */
  public Path settingsLock() {
    return root.resolve("settings.lock");
  }

  public Path state() {
    return root.resolve("state.json");
  }

  public Path status() {
    return root.resolve("status.json");
  }

  /** 制御の要求を置くディレクトリ。 */
  public Path control() {
    return root.resolve("control");
  }

  /** 段とジョブを回す JVM が持つロック。 */
  public Path trainLock() {
    return root.resolve(".train.lock");
  }

  /** 反復ごとの作業ディレクトリを並べるディレクトリ。 */
  public Path work() {
    return root.resolve("work");
  }

  /** 反復の作業ディレクトリ（段の途中の出力と完了マーカーを置く）。 */
  public Path work(int iteration) {
    return work().resolve(number(iteration));
  }

  /** 反復に固定した設定の行と jar を書くファイル。 */
  public Path iterationFile(int iteration) {
    return work(iteration).resolve("ITERATION.json");
  }

  /** 確定したコホート。 */
  public Path rows(int iteration) {
    return root.resolve("rows").resolve(number(iteration));
  }

  public Path snapshots() {
    return root.resolve("snapshots");
  }

  /** 公開したバージョンの重み。 */
  public Path snapshot(int version) {
    return snapshots().resolve(number(version));
  }

  public Path grp() {
    return root.resolve("grp");
  }

  public Path jobs() {
    return root.resolve("jobs");
  }

  public Path job(String name) {
    return jobs().resolve(name);
  }

  /** 王者の保存先。 */
  public Path champion() {
    return root.resolve("champion");
  }

  public Path reports() {
    return root.resolve("reports");
  }

  /** reports の中のファイル。 */
  public Path report(String name) {
    return reports().resolve(name);
  }

  /** バージョンと卓から作るジョブの名前（{@code v00824-x1n3}）。 */
  public static String jobName(int version, String table) {
    return "v" + number(version) + "-" + table;
  }

  /**
   * ジョブの名前から、バージョンと卓を読む。
   *
   * @return 名前がこの形でなければ空
   */
  public static Optional<JobName> parseJobName(String name) {
    Matcher matcher = JOB.matcher(name);
    if (!matcher.matches()) {
      return Optional.empty();
    }
    try {
      return Optional.of(new JobName(Integer.parseInt(matcher.group(1)), matcher.group(2)));
    } catch (NumberFormatException overflow) {
      return Optional.empty();
    }
  }

  /** ジョブの名前を、バージョンと卓に分けたもの。 */
  public record JobName(int version, String table) {}

  private static String number(int value) {
    return String.format("%05d", value);
  }
}
