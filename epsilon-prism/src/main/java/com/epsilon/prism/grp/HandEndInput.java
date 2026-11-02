package com.epsilon.prism.grp;

/**
 * 半荘が続く局末 1 つの Φ の入力。
 *
 * @param prefix 次局 root の公開系列（GRP の入力。書き換えない）
 * @param next 次局の整数（点数は精算後。点数だけの Φ はこれだけを読む）
 */
public record HandEndInput(float[] prefix, HandStart next) {}
