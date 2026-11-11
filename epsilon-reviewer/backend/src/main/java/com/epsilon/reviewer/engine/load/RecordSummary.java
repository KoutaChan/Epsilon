package com.epsilon.reviewer.engine.load;

import java.util.List;

/**
 * 牌譜の概要。読み込み、履歴、解析結果のどれでも同じ形で返す。
 *
 * @param source 牌譜の形式（tenhou・mjai・majsoul）
 * @param names 席ごとの名前。要素の数は席の数（4 人か 3 人）で、名前の無い席は空の文字列
 * @param finalScores 席ごとの終局時の持ち点。読み込んだ牌譜では必ずある。前のバージョンが保存した結果の概要では null のことがある
 * @param roundCount 局の数
 */
public record RecordSummary(
    String source, List<String> names, List<Integer> finalScores, int roundCount) {}
