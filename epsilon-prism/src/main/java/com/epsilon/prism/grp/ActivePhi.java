package com.epsilon.prism.grp;

import com.epsilon.prism.collect.HandValue;

/**
 * 反復 t の収集で使う Φ。
 *
 * @param entry 使う項目（iterations.jsonl の phi とコホートの記録の先頭に書く）
 * @param value 学習の JVM の局末の値（閉じるのは持ち主の {@code TrainingResources}）
 * @param resetsLedger 反復 t − 1 の項目が点数だけの Φ で反復 t の項目が GRP なら true（運の精算の R² の持ち越しを 0 にし、K′ = K にする）。
 *     phi.jsonl の 2 つの項目で決め、開いていた項目に依らない（再開でも同じ）
 */
public record ActivePhi(PhiEntry entry, HandValue value, boolean resetsLedger) {}
