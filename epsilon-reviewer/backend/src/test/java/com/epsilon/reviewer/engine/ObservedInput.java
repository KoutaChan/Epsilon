package com.epsilon.reviewer.engine;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * テストで比べる O のフィールド。方策が受け取った O を読み直して作る。席は絶対席で書く。イベントは 1 つずつ文字列にする。打牌は
 * 「discard 席 種類 赤 ツモ切り 立直 鳴いた席」、面子は「meld 席 面子」、北抜きは「nuki 席」と書く。
 */
record ObservedInput(
    EngineDecisionKind kind,
    List<String> events,
    List<Integer> hand,
    int handAka,
    List<Integer> scores,
    List<String> riichi,
    int honba,
    int kyotaku,
    List<Integer> indicators,
    List<Integer> legal,
    boolean furiten,
    int drawn) {
  static ObservedInput of(Variant variant, long[] words) {
    Observation o = new Observation(variant).bind(words, 0);
    int seats = variant.seats();
    List<String> events = new ArrayList<>();
    for (int event = 0; event < o.events(); event++) {
      int author = o.absoluteSeat(o.eventAuthor(event));
      if (o.eventKind(event) == EventKind.DISCARD) {
        int calledBy = o.discardCalledBy(event);
        events.add(
            String.join(
                " ",
                "discard",
                Integer.toString(author),
                Integer.toString(o.discardTile(event)),
                Boolean.toString(o.discardAka(event)),
                Boolean.toString(o.discardTsumogiri(event)),
                Boolean.toString(o.discardRiichi(event)),
                Integer.toString(calledBy < 0 ? -1 : o.absoluteSeat(calledBy))));
      } else if (o.eventKind(event) == EventKind.NUKI) events.add("nuki " + author);
      else events.add("meld " + author + " " + o.meld(event));
    }
    int[] counts = new int[Tiles.KINDS];
    o.handCounts(counts);
    List<Integer> scores = new ArrayList<>();
    List<String> riichi = new ArrayList<>();
    for (int seat = 0; seat < seats; seat++) {
      int relative = Math.floorMod(seat - o.observer(), seats);
      scores.add(o.score(relative));
      riichi.add(o.riichi(relative).name());
    }
    List<Integer> indicators = new ArrayList<>();
    for (int index = 0; index < o.indicators(); index++) indicators.add(o.indicatorTile(index));
    List<Integer> legal = new ArrayList<>();
    for (int index = 0; index < o.legalActions(); index++) legal.add(o.action(index).code());
    return new ObservedInput(
        o.decisionKind(),
        List.copyOf(events),
        Arrays.stream(counts).boxed().toList(),
        o.handAkaMask(),
        List.copyOf(scores),
        List.copyOf(riichi),
        o.honba(),
        o.kyotaku(),
        List.copyOf(indicators),
        List.copyOf(legal),
        o.furitenTemporary(),
        o.drawnTile());
  }
}
