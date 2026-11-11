package com.epsilon.classic.input;

import com.epsilon.classic.ArchitectureScoped;

/**
 * 入力のフィールド。各定数は要素の数を持つ。オフセットは、アーキテクチャが持つ定数を宣言の順に並べ、要素の数を足し上げて求める。
 */
public interface InputField extends ArchitectureScoped {

  /** フィールドの要素の数。 */
  default int count() {
    return 1;
  }
}
