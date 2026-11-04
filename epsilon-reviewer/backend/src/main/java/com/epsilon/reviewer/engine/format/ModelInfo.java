package com.epsilon.reviewer.engine.format;

/** 解析結果・履歴・解析の始まりに書くモデルの表示。 */
public record ModelInfo(String modelId, String revision, String displayName, String version) {}
