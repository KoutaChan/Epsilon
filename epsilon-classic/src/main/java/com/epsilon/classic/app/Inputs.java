package com.epsilon.classic.app;

import com.epsilon.agent.settings.SettingsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** コマンドが読むパスと書くパスを確かめる。設定を引数に結び付けるときには見ないので、コマンドの始めに確かめ、違えば使い方の誤りにする。 */
final class Inputs {

  private Inputs() {}

  /**
   * @param directory 確かめるパス
   * @param argument 引数の名前（エラーメッセージに書く）
   * @throws SettingsException パスがディレクトリでないとき
   */
  static void requireDirectory(Path directory, String argument) {
    if (!Files.isDirectory(directory)) {
      throw new SettingsException(List.of("Argument --" + argument + " names no directory: " + directory));
    }
  }

  /**
   * @param out 書く先
   * @param what 書く物の名前（エラーメッセージに書く）
   * @throws SettingsException 書く先が既にあるとき
   */
  static void requireNew(Path out, String what) {
    if (Files.exists(out)) {
      throw new SettingsException(List.of("The " + what + " already exists: " + out));
    }
  }
}
