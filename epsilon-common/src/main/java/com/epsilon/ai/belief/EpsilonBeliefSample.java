package com.epsilon.ai.belief;

/**
 * 公開局面を表す系列固有の入力と、全情報から作ったBelief教師データの組。
 *
 * @param input 1局面だけを保持する系列が所有する型付き入力
 * @param target 同じ局面の非公開情報から作った Belief 教師
 */
public record EpsilonBeliefSample<I>(I input, EpsilonBeliefTarget target) {}
