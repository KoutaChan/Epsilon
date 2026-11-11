package com.epsilon.reviewer.model;

/**
 * カタログに載せるモデル 1 つ。使えない理由はログに残し、画面には送らない。
 *
 * @param seats モデルが読む牌譜の人数（4 か 3）
 */
public record CatalogModel(
    String modelId,
    String revision,
    String displayName,
    String version,
    int seats,
    boolean available) {}
