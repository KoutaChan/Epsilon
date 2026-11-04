package com.epsilon.reviewer;

import java.nio.file.Path;

/**
 * 組み立ての設定。
 *
 * @param data 保存先
 * @param models models.json。無ければ null（空のカタログ）
 * @param maxOpenModels 同時に開くモデルの数
 */
public record ReviewerConfig(Path data, Path models, int maxOpenModels) {}
