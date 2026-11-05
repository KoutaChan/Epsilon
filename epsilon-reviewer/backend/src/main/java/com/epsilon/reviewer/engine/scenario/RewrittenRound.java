package com.epsilon.reviewer.engine.scenario;

import com.epsilon.replay.ReplayEvent;
import com.epsilon.reviewer.engine.view.TableState;
import java.util.List;

/**
 * 変更を写した判断の局。
 *
 * @param events 局の出来事（局の始めから）。判断より前は判断の席の目で成り立ち、その席に見える牌の物理牌は重ならない。他家の伏せた配牌と
 *     ツモは記録のままなので、変更が割り当てた物理牌と重なることがあり、替えた他家の打牌はその席の手に入っていない。判断より後は記録のまま
 * @param table 変更を写した判断の時点の表示の局面（判断の席の手牌は並べ直す）
 */
public record RewrittenRound(List<ReplayEvent> events, TableState table) {}
