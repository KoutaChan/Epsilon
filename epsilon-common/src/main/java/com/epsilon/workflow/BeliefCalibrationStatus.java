package com.epsilon.workflow;

import com.google.gson.annotations.SerializedName;
import java.util.Locale;

/** Belief較正処理の完了状態。 */
public enum BeliefCalibrationStatus {
  @SerializedName("calibrated")
  CALIBRATED,
  @SerializedName("checkpoint-not-found")
  CHECKPOINT_NOT_FOUND,
  @SerializedName("no-supported-log-files")
  NO_SUPPORTED_LOG_FILES,
  @SerializedName("no-belief-samples")
  NO_BELIEF_SAMPLES;

  /** 状態名を小文字にし、単語をハイフンで区切ったCLI用の表記を返す。 */
  @Override
  public String toString() {
    return name().toLowerCase(Locale.ROOT).replace('_', '-');
  }
}
