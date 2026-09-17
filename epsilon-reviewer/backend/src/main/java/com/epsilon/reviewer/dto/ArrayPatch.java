package com.epsilon.reviewer.dto;

import java.util.List;

/** 指定位置から指定個数の要素を除き、その位置に置換後の要素を挿入する配列差分。差分自体がnullなら変更しない。 */
public record ArrayPatch<T>(int index, int removeCount, List<T> values) {}
