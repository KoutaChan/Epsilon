package com.epsilon.spi;

import java.nio.file.Path;
import java.util.Map;

/** 系列が所有するチェックポイントと入力仕様から推論実行環境を開く。 */
public interface ModelProvider {
  String seriesId();

  BatchedPolicy open(
      Path checkpoint, Map<String, String> options, PolicyExecutionContext execution);
}
