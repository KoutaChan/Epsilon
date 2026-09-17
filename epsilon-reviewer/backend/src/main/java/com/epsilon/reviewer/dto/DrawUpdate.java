package com.epsilon.reviewer.dto;

/** ツモ牌の表示を更新する差分。{@code tile == null}ならツモ牌の表示を消し、この差分自体がnullなら変更しない。 */
public record DrawUpdate(String tile) {}
