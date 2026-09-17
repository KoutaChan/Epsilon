package com.epsilon.ai.grp;

import java.util.Locale;
import java.util.Objects;

/**
 * 固定した GRP 教師モデルを世代番号だけでなく推論用ファイルの SHA-256 で一意に表す。
 *
 * @param iteration 教師モデルチェックポイントの世代番号。無効時は {@code -1}
 * @param checkpointSha256 チェックポイント内容の SHA-256。無効時は空文字列
 */
public record EpsilonGrpTeacherIdentity(int iteration, String checkpointSha256) {

  private static final String SHA256_PATTERN = "[0-9a-f]{64}";

  /**
   * 教師モデルの有効・無効表現と SHA-256 形式を検証する。
   *
   * @param iteration 教師モデルチェックポイントの世代番号。無効時は {@code -1}
   * @param checkpointSha256 チェックポイント内容の小文字 SHA-256。無効時は空文字列
   */
  public EpsilonGrpTeacherIdentity {
    checkpointSha256 = Objects.requireNonNull(checkpointSha256, "checkpointSha256");
    if (iteration == -1) {
      if (!checkpointSha256.isEmpty()) {
        throw new IllegalArgumentException("Disabled GRP teacher must have an empty SHA-256");
      }
    } else if (iteration < 0 || !checkpointSha256.matches(SHA256_PATTERN)) {
      throw new IllegalArgumentException(
          "Enabled GRP teacher requires iteration >= 0 and lowercase SHA-256: iteration="
              + iteration
              + " sha256="
              + checkpointSha256);
    }
  }

  /**
   * GRP 教師モデルを使用しない識別情報を返す。
   *
   * @return 世代番号 {@code -1} の無効状態を表す識別情報
   */
  public static EpsilonGrpTeacherIdentity disabled() {
    return new EpsilonGrpTeacherIdentity(-1, "");
  }

  /**
   * 固定したチェックポイントを指す有効な教師モデル識別情報を作る。
   *
   * @param iteration 教師モデルチェックポイントの世代番号
   * @param checkpointSha256 推論用ファイルの SHA-256
   * @return 正規化済み識別情報
   */
  public static EpsilonGrpTeacherIdentity enabled(int iteration, String checkpointSha256) {
    return new EpsilonGrpTeacherIdentity(
        iteration,
        Objects.requireNonNull(checkpointSha256, "checkpointSha256").toLowerCase(Locale.ROOT));
  }

  /**
   * 識別情報が実在する固定した教師モデルを指すかを返す。
   *
   * @return 教師モデルが有効なら {@code true}
   */
  public boolean enabled() {
    return iteration >= 0;
  }
}
