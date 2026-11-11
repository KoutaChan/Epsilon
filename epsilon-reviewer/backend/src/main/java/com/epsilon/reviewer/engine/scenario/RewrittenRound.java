package com.epsilon.reviewer.engine.scenario;

import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.reviewer.engine.view.TableState;
import java.util.List;

/**
 * もしもの変更を反映した、判断のある局。判断する席から見える卓を、この局の始まりの状態と配牌、最初のドラ表示牌で始めて、
 * イベントを順に適用すると判断の時点に着く。
 *
 * @param round 局の通し番号
 * @param honba 局の始まりの本場
 * @param kyotaku 局の始まりの供託の本数
 * @param scores 局の始まりの持ち点（絶対席の順）
 * @param hand 判断する席の配牌（物理牌の id 13 枚）
 * @param indicator 最初のドラ表示牌の物理牌の id
 * @param events 判断する席から見た、判断の時点までのイベント（他家のツモ牌は伏せる）。判断する席に見える牌どうしで物理牌は重ならない
 * @param table 変更を反映した、判断の時点の表示用の局面（判断する席の手牌は並べ直す）
 */
public record RewrittenRound(
    int round,
    int honba,
    int kyotaku,
    int[] scores,
    int[] hand,
    int indicator,
    List<RecordEvent> events,
    TableState table) {}
