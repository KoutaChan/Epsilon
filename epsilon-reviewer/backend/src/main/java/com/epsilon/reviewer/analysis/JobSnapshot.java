package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.error.ErrorBody;

/**
 * 解析の進み。
 *
 * @param progress 0..100。completed になるまで 99 を超えない
 * @param resultId completed のときだけ。他は null
 * @param error failed のときだけ。他は null
 */
public record JobSnapshot(
    String jobId, JobStatus status, int progress, String resultId, ErrorBody error) {}
