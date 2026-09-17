package com.epsilon.arena;

import java.nio.file.Path;
import java.util.Map;

/** 系列名とパスを別々に保持する。Windows のドライブ記号もそのまま扱う。 */
public record ModelSpec(String series, Path checkpoint, Map<String, String> options) {
  public ModelSpec {
    options = Map.copyOf(options);
  }
}
