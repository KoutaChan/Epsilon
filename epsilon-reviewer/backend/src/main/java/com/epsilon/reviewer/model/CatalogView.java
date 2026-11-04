package com.epsilon.reviewer.model;

import java.util.List;

/**
 * 画面に出すカタログ（models.json の順）。
 *
 * @param defaultModel 既定のモデル。無ければ null
 */
public record CatalogView(List<CatalogModel> models, ModelKey defaultModel) {}
