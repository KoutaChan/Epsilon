package com.epsilon.runtime;

import com.google.gson.annotations.SerializedName;

/** 計測値の可用性。値がない場合と、一部のデバイスだけ計測できた場合を区別する。 */
public enum MeasurementStatus {
  @SerializedName("available")
  AVAILABLE,
  @SerializedName("unavailable")
  UNAVAILABLE,
  @SerializedName("not-applicable")
  NOT_APPLICABLE,
  @SerializedName("partial")
  PARTIAL
}
