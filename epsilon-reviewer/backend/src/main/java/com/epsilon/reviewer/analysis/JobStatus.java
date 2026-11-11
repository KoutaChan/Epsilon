package com.epsilon.reviewer.analysis;

import com.google.gson.annotations.SerializedName;

/** 解析ジョブの状態。queued → running → saving → completed と進むか、failed か cancelled で終わる。 */
public enum JobStatus {
  @SerializedName("queued")
  QUEUED,
  @SerializedName("running")
  RUNNING,
  /** 保存の途中。ここまで進むと、取り消しは効かない。 */
  @SerializedName("saving")
  SAVING,
  @SerializedName("completed")
  COMPLETED,
  @SerializedName("failed")
  FAILED,
  @SerializedName("cancelled")
  CANCELLED;

  boolean finished() {
    return this == COMPLETED || this == FAILED || this == CANCELLED;
  }
}
