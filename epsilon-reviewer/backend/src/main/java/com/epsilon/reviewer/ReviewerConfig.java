package com.epsilon.reviewer;

import com.epsilon.reviewer.model.ModelFamilies;
import java.nio.file.Path;

/**
 * Reviewer を組み立てるときの設定。
 *
 * @param data 解析結果の保存先
 * @param models models.json。null なら空のカタログで始める
 * @param settings モデルの上限と解析ジョブの設定
 * @param families モデルの系列ごとの方策の実装
 */
public record ReviewerConfig(Path data, Path models, ReviewerSettings settings, ModelFamilies families) {}
