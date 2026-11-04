package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.engine.format.ModelInfo;

/** 受け付けた解析。 */
public record AnalysisStarted(String jobId, ModelInfo model) {}
