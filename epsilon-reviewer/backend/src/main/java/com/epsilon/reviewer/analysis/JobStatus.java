package com.epsilon.reviewer.analysis;

import com.google.gson.annotations.SerializedName;

/** 解析の仕事の状態。queued → running → saving → completed、または failed・cancelled。 */
public enum JobStatus {
  @SerializedName("queued")
  QUEUED,
  @SerializedName("running")
  RUNNING,
  /** 保存の途中（この後の取り消しは効かない）。 */
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
