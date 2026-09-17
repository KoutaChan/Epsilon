package com.epsilon.reviewer.dto;

/** 解析に実際に使ったモデルの不変な識別情報。 */
public record ModelInfo(
    String modelId, String revision, String displayName, String version, String series) {}
