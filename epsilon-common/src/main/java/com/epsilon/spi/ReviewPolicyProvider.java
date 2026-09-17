package com.epsilon.spi;

import java.nio.file.Path;
import java.util.Map;

/** モデル系列ごとのチェックポイントを読み込み、牌譜解析用の方策インターフェースへ接続する。 */
public interface ReviewPolicyProvider {
  String seriesId();

  ReviewPolicy openReview(Path checkpoint, Map<String, String> options);
}
