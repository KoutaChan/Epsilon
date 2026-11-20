package com.epsilon.reviewer.engine.scenario;

import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.epsilon.reviewer.engine.view.TableView;

/**
 * もしもの評価の結果。候補は解析の判断と同じ形で返し、牌譜の手がまだ合法なら {@code chosen} がその位置を指す。
 *
 * @param table 鳴いた後の局面。鳴かないときは null（画面が変更から局面を作る）
 */
public record ScenarioAnswer(EvaluatedDecision decision, TableView table) {}
