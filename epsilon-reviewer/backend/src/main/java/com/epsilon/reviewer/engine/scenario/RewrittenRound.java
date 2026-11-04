package com.epsilon.reviewer.engine.scenario;

import com.epsilon.replay.ReplayEvent;
import com.epsilon.reviewer.engine.view.TableState;
import java.util.List;

/**
 * 変更を写した判断の局。
 *
 * @param events 局の出来事（局の始めから）
 * @param table 変更を写した判断の時点の表示の局面（判断の席の手牌は並べ直す）
 */
public record RewrittenRound(List<ReplayEvent> events, TableState table) {}
