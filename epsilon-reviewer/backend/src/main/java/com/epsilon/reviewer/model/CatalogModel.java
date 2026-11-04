package com.epsilon.reviewer.model;

/** カタログの 1 モデル。使えない理由は記録に残し、画面には送らない。 */
public record CatalogModel(
    String modelId, String revision, String displayName, String version, boolean available) {}
