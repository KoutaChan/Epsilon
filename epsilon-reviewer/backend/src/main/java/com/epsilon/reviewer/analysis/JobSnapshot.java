package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.error.ErrorBody;

/**
 * 解析ジョブの進み具合。
 *
 * @param progress 進み具合（0〜100 %）。completed になるまでは 99 を超えない
 * @param resultId completed のときだけ入る。ほかは null
 * @param error failed のときだけ入る。ほかは null
 */
public record JobSnapshot(
    String jobId, JobStatus status, int progress, String resultId, ErrorBody error) {}
