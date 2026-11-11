package com.epsilon.classic.input;

/** 整数の値を持つ入力フィールド。値は 0 以上、幅未満。 */
public interface CategoricalField extends InputField {

  /** 取りうる値の数。埋め込むフィールドでは、埋め込みの表でこのフィールドが占める行の数。 */
  int width();
}
