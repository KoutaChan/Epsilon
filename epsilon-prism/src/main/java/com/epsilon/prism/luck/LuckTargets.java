package com.epsilon.prism.luck;

import ai.djl.ndarray.NDArray;

/**
 * 運の網 Ω の 1 塊の目標（デバイスの上）。網の入力ではない。
 *
 * @param pool 正準の結果の順のプールの枚数 [行, 37]（FP32）
 * @param chosen 実際の結果の平らな番号 [行]（INT64、行 × 37 + 正準の結果）
 * @param y 局末の Φ [行, 4]（FP32、行為者から見た相対席の順）
 * @param weight 重み [行]（FP32）
 */
record LuckTargets(NDArray pool, NDArray chosen, NDArray y, NDArray weight) {}
