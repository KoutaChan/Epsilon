package com.epsilon.classic.input;

/** 実数の値を持つフィールド。書く値は、量を尺度で割ったもの。 */
public interface NumericField extends InputField {

  /** 量を割る尺度。 */
  float scale();
}
