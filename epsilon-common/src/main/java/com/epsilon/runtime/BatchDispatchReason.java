package com.epsilon.runtime;

/** 最終バッチの発行を決めた理由です。 */
public enum BatchDispatchReason {
  FULL,
  SUPPLY,
  DEADLINE
}
