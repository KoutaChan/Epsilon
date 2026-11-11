package com.epsilon.reviewer.analysis;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;

/**
 * 解析ジョブの設定。終わっていないジョブはオーナーごとに 1 つまで。
 *
 * @param workers 解析を実行するスレッドの数
 * @param queue 実行を待たせておけるジョブの数
 * @param keepHours 終わったジョブを覚えておく時間（時間）
 */
public record AnalysisSettings(
    @Key(doc = "解析を実行するスレッドの数", value = "1") @Range(min = 1) int workers,
    @Key(doc = "実行を待たせておけるジョブの数", value = "16") @Range(min = 1) int queue,
    @Key(doc = "終わったジョブを覚えておく時間（時間）", value = "24") @Range(min = 1) int keepHours) {}
