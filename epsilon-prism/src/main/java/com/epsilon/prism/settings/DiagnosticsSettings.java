package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Key;
import com.epsilon.prism.hip.HipResources;

/**
 * HIP のランタイムの検査を設定する。
 *
 * @param hipChecked 検査つきのビルドを使うか
 */
public record DiagnosticsSettings(
    @Key(doc = "HIP のランタイムを検査つきのビルドにするか", value = "false") boolean hipChecked) {

  /** ネットワークを開く前に、読み込むライブラリを設定する。 */
  public void apply() {
    System.setProperty(HipResources.CHECKED_PROPERTY, Boolean.toString(hipChecked));
  }
}
