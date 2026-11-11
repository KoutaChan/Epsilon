package com.epsilon.agent.settings;

import java.util.Objects;

/**
 * 設定の record の検査が見つけたエラー。どのコンポーネントの値が悪いかを持つ。結び付ける側が、そのキーのパス・値・書かれた場所を
 * 添えて知らせる。投げるのは record のコンパクトコンストラクタ（セクションをまたぐ検査なら、全体の record）。
 */
public final class InvalidSetting extends IllegalArgumentException {

  private final String component;

  /**
   * @param component 悪い値のコンポーネントの名前。入れ子のセクションのコンポーネントは点でつなぐ（{@code phi.grp}）
   * @param message 英語の 1 文
   */
  public InvalidSetting(String component, String message) {
    super(message);
    this.component = Objects.requireNonNull(component, "component");
  }

  /** 悪い値のコンポーネントの名前（点でつないだパス）。 */
  public String component() {
    return component;
  }
}
