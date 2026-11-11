package com.epsilon.reviewer.engine.scenario;

/**
 * もしもで評価し直す 1 つの判断。
 *
 * @param round 解析結果の {@code rounds} の何番目か
 * @param eventIndex 判断を決める手順の位置（文書の resolve）
 * @param seat 判断する席
 * @param call 応答の判断で鳴いたとするチー・ポンの行動の番号。鳴かなければ null
 */
public record ScenarioQuery(
    int round, int eventIndex, int seat, ScenarioEdits edits, Integer call) {}
