package com.epsilon.config.settings;

/**
 * RiichiLabへの接続と自動対局の設定。チェックポイントの既定パスは各系列の同梱設定が所有する。
 *
 * @param token 接続トークン。対局コマンドの起動時だけ必須
 * @param checkpointDir 対局用チェックポイントを検索するディレクトリ、または採用済みモデルの保存先
 * @param maxGames 完了する対局数。0なら停止するまで次の対局へ参加する
 */
@SettingsPrefix("epsilon.riichi")
public record RiichiSettings(
    @Default("") String token, String checkpointDir, @Default("0") @NonNegative int maxGames) {

  /** 接続トークンを設定の診断出力へ含めない。 */
  @Override
  public String toString() {
    return "RiichiSettings[token=<redacted>, checkpointDir="
        + checkpointDir
        + ", maxGames="
        + maxGames
        + "]";
  }
}
