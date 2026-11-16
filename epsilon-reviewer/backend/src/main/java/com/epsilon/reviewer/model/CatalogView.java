package com.epsilon.reviewer.model;

import java.util.List;

/**
 * 画面に出すカタログ（models.json の順）。
 *
 * @param defaultModels ルールごとの既定のモデル（models.json の順、1 つのルールに 1 つまで）。無ければ空
 */
public record CatalogView(List<CatalogModel> models, List<ModelKey> defaultModels) {}
