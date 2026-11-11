package com.epsilon.reviewer.engine.format;

/** 解析結果・履歴・解析を受け付けた応答に書く、モデルの表示用の情報。 */
public record ModelInfo(String modelId, String revision, String displayName, String version) {}
