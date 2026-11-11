package com.epsilon.classic.campaign;

import java.util.Objects;

/**
 * 収集の段の報告。
 *
 * @param version 学習器が打ったバージョン
 * @param games 半荘の数
 * @param decisions 学習器の席の判断の数
 * @param champion 収集のときの王者の名前
 * @param pool 相手に選べた過去の王者の数
 * @param opponent この macro の相手の名前
 * @param bytes 確定したコホートのファイルの大きさの和（バイト）
 */
public record CollectReport(
    int version, int games, long decisions, String champion, int pool, String opponent, long bytes) {

  public CollectReport {
    Objects.requireNonNull(champion, "champion");
    Objects.requireNonNull(opponent, "opponent");
  }
}
