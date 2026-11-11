package com.epsilon.agent.play;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;

/** 卓の設定のセクション。自己対局と評価の卓が半荘を進めるときに読む。どのキーも速さだけを変える。結果の分布は変わらないが、ビット単位では変わりうる。 */
public record TableSettings(
    @Key(doc = "同時に進める半荘の数", value = "8192", change = Change.MACHINE) @Range(min = 1) int activeGames,
    @Key(doc = "半荘を進めるスレッドの数", value = "28", change = Change.MACHINE) @Range(min = 1) int workers,
    @Key(doc = "推論のバッチで、最も古い行を待たせてよい時間の上限（µs）", value = "100000", change = Change.MACHINE) @Range(min = 0)
        long batchWaitMicros,
    @Key(doc = "行を書くスレッドの数", value = "4", change = Change.MACHINE) @Range(min = 1) int writers) {}
