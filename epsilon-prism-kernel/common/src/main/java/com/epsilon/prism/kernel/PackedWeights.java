package com.epsilon.prism.kernel;

import com.epsilon.prism.kernel.ModelSchema.Model;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * 収集の計画に公開する重みの束。配布済みの pack のバイトの並びを、その重みを作った宣言と一緒に渡す。
 *
 * @param model 重みを作ったモデルの宣言
 * @param generation 重みの世代（0 以上）
 * @param data ヘッダーつきの pack（direct）
 * @param options 計画を作るときに決める候補の重みとマスク
 * @param restoration 候補のコードを入力から復元するときの列の並び（復元しないなら空）
 */
public record PackedWeights(
    Model model,
    long generation,
    ByteBuffer data,
    CollectionPlanOptions options,
    int[] restoration) {
  public PackedWeights {
    Objects.requireNonNull(model);
    Objects.requireNonNull(options);
    if (generation < 0 || !data.isDirect() || data.remaining() < PackedWeightLayout.HEADER_BYTES)
      throw new IllegalArgumentException("Invalid weight pack or generation");
    data = data.slice().asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
    restoration = restoration.clone();
  }

  /** 候補のコードの復元の並び。呼び出し側は書き換えない。 */
  @Override
  public int[] restoration() {
    return restoration;
  }

  @Override
  public ByteBuffer data() {
    return data.duplicate().order(ByteOrder.LITTLE_ENDIAN);
  }
}
