package com.epsilon.agent.policy;

import java.nio.file.Path;
import java.util.Objects;

/**
 * 打ち手の系列。名前は、Reviewer の models.json で series に書く値。接頭辞は、卓の相手の書き方（接頭辞:チェックポイント）と、
 * 評価結果の名前の先頭に使う。名前も接頭辞も保存した設定と結果に残るので、変えない。
 */
public enum Series {
  PRISM("epsilon-prism", "prism", true),
  NANO("epsilon-nano", "nano", true),
  MAJOR("epsilon", "major", true),
  RULE_BOT("rulebot", "heuristic", false);

  /** 卓の相手の書き方で、接頭辞とチェックポイントのパスを分ける文字。 */
  private static final char SEPARATOR = ':';

  private static final Series[] VALUES = values();

  private final String id;
  private final String prefix;
  private final boolean checkpointed;

  Series(String id, String prefix, boolean checkpointed) {
    this.id = id;
    this.prefix = prefix;
    this.checkpointed = checkpointed;
  }

  public String id() {
    return id;
  }

  public String prefix() {
    return prefix;
  }

  /** チェックポイントを開く系列なら true（{@link #RULE_BOT} は開かない）。 */
  public boolean checkpointed() {
    return checkpointed;
  }

  /**
   * 評価結果の行に付ける名前（{@link Policy#label}）を作る。チェックポイントの有無が系列と合わないか、パスにディレクトリ名が無ければ
   * IllegalArgumentException を投げる。
   *
   * @param checkpoint チェックポイントのディレクトリ（チェックポイントを取らない系列は null）
   * @return 接頭辞、":"、チェックポイントのディレクトリ名をつないだ名前。チェックポイントを取らない系列は接頭辞だけ
   */
  public String label(Path checkpoint) {
    requireCheckpoint(checkpoint);
    if (!checkpointed) {
      return prefix;
    }
    Path name = checkpoint.getFileName();
    if (name == null) {
      throw new IllegalArgumentException("Checkpoint has no directory name: " + checkpoint);
    }
    return prefix + SEPARATOR + name;
  }

  /**
   * 名前（models.json の series に書く値）から系列を引く。
   *
   * @throws IllegalArgumentException どの系列の名前でもないとき
   */
  public static Series ofId(String id) {
    for (Series series : VALUES) {
      if (series.id.equals(id)) {
        return series;
      }
    }
    throw new IllegalArgumentException("No series has the id " + id);
  }

  /**
   * 接頭辞（卓の相手の書き方の先頭）から系列を引く。
   *
   * @throws IllegalArgumentException どの系列の接頭辞でもないとき
   */
  public static Series ofPrefix(String prefix) {
    for (Series series : VALUES) {
      if (series.prefix.equals(prefix)) {
        return series;
      }
    }
    throw new IllegalArgumentException("No series has the prefix " + prefix);
  }

  /**
   * 卓の相手の書き方を読む。チェックポイントを取る系列は「接頭辞:チェックポイントのパス」と書き、最初の ":" より後ろが全てパスになる。
   * チェックポイントを取らない系列は、接頭辞だけを書く。
   *
   * @param spec 相手の書き方（heuristic、nano:/models/nano/ppo17 など）
   * @return 相手の系列とチェックポイント
   * @throws IllegalArgumentException 接頭辞を知らないとき、パスが空のとき、チェックポイントの有無が系列と合わないとき
   */
  public static Spec parse(String spec) {
    int separator = Objects.requireNonNull(spec, "spec").indexOf(SEPARATOR);
    if (separator < 0) {
      return new Spec(ofPrefix(spec), null);
    }
    Series series = ofPrefix(spec.substring(0, separator));
    String checkpoint = spec.substring(separator + 1);
    if (checkpoint.isEmpty()) {
      throw new IllegalArgumentException("The opponent " + spec + " names no checkpoint");
    }
    return new Spec(series, Path.of(checkpoint));
  }

  private void requireCheckpoint(Path checkpoint) {
    if (checkpointed != (checkpoint != null)) {
      throw new IllegalArgumentException(
          this + (checkpointed ? " needs a checkpoint" : " takes no checkpoint") + ": " + checkpoint);
    }
  }

  /**
   * 卓の相手。{@link #parse} が相手の書き方から作る。チェックポイントの有無が系列と合わなければ、IllegalArgumentException を投げる。
   *
   * @param checkpoint チェックポイントのディレクトリ（チェックポイントを取らない系列は null）
   */
  public record Spec(Series series, Path checkpoint) {

    public Spec {
      Objects.requireNonNull(series, "series").requireCheckpoint(checkpoint);
    }
  }
}
