package com.epsilon.config.settings;

/**
 * 整数のカテゴリ入力と浮動小数点の数値入力を、JavaからDJL/PyTorchへ転送する方式。
 *
 * <p>ホスト側で再利用する転送バッファの種類を指定する。転送先のテンソルは呼び出し元のメモリ管理オブジェクトが所有し、学習と自己対戦のどちらでも呼び出し単位で解放する。
 */
public enum DecisionTensorTransfer {
  /** ページ固定したホストバッファを使います。GPU への H2D 転送を優先する通常経路です。 */
  PINNED_BUFFER,

  /** Java ダイレクトバッファを再利用します。ページ固定メモリを使わない比較・退避用経路です。 */
  DIRECT_BUFFER
}
