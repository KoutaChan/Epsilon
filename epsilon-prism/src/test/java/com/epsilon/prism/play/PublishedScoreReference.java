package com.epsilon.prism.play;

import java.io.IOException;
import java.util.Properties;

/** 配布モデルの実行経路に対応する合法手スコアの参照値を読む。 */
final class PublishedScoreReference {
  private PublishedScoreReference() {}

  static String resource(String backend) {
    return switch (backend) {
      case "EAGER", "HIP" -> "published-scores.properties";
      case "CUDA" -> "published-cuda-scores.properties";
      default -> throw new IllegalArgumentException("Unknown score backend: " + backend);
    };
  }

  static Properties load(String backend) throws IOException {
    String name = resource(backend);
    try (var input = PublishedScoreReference.class.getResourceAsStream(name)) {
      if (input == null) throw new IOException("Missing published score reference: " + name);
      var scores = new Properties();
      scores.load(input);
      return scores;
    }
  }
}
