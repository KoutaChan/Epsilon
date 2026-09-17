package com.epsilon.ai.grp;

/** GRPチェックポイントの形式、モデル構成、累計更新回数、世代番号を記録する保存情報。 */
public final class EpsilonGrpCheckpointBundle {

  /** チェックポイントの保存情報の形式バージョン。 */
  public int checkpointVersion;

  /** 保存時の GRPのモデル構成の識別子。 */
  public String architecture;

  /** オプティマイザーの累計更新回数。 */
  public int globalStep;

  /** GRP 世代番号。 */
  public int iteration;

  /** JSON 復元用の空の保存情報を作る。 */
  public EpsilonGrpCheckpointBundle() {}

  /**
   * 現行のモデル構成を表すチェックポイントの保存情報を作る。
   *
   * @param globalStep オプティマイザーの累計更新回数
   * @param iteration GRP 世代番号
   */
  public EpsilonGrpCheckpointBundle(int globalStep, int iteration) {
    this(globalStep, iteration, EpsilonGrpNetwork.DEFAULT_HIDDEN, EpsilonGrpNetwork.DEFAULT_LAYERS);
  }

  public EpsilonGrpCheckpointBundle(int globalStep, int iteration, int hidden, int layers) {
    this.checkpointVersion = 2;
    this.architecture = EpsilonGrpNetwork.architectureSummary(hidden, layers);
    this.globalStep = globalStep;
    this.iteration = iteration;
  }

  /** 保存時の構成を読み、入力と数式の契約も一致することを検証する。 */
  public Configuration configuration() throws java.io.IOException {
    var match =
        java.util.regex.Pattern.compile(".* gruHidden=(\\d+) layers=(\\d+) .*")
            .matcher(architecture == null ? "" : architecture);
    if (!match.matches())
      throw new java.io.IOException("Unsupported GRP architecture: " + architecture);
    try {
      int hidden = Integer.parseInt(match.group(1));
      int layers = Integer.parseInt(match.group(2));
      if (hidden <= 0
          || layers <= 0
          || !EpsilonGrpNetwork.architectureSummary(hidden, layers).equals(architecture))
        throw new java.io.IOException("Unsupported GRP architecture: " + architecture);
      return new Configuration(hidden, layers);
    } catch (NumberFormatException e) {
      throw new java.io.IOException("Invalid GRP architecture: " + architecture, e);
    }
  }

  public record Configuration(int hidden, int layers) {}
}
