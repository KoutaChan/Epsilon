package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.engine.format.ModelInfo;

/** 解析ジョブを受け付けたときの応答。 */
public record AnalysisStarted(String jobId, ModelInfo model) {}
