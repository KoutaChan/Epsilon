package com.epsilon.agent.observe;

import com.epsilon.agent.observe.ObservationSchema.DiscardBits;
import com.epsilon.agent.observe.ObservationSchema.EventBits;
import com.epsilon.agent.observe.ObservationSchema.MeldBits;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.EventKind;

/**
 * エンジンが記録した公開のイベント（EventLog）を、番号の順に、記録 O のイベントのセクションへ書く。これまでの経過を河や手から
 * 組み立て直さず、エンジンの記録を前から 1 回だけ読む。
 *
 * <p>席は観測席から見た相対席で書く。鳴いた席も相対席で書き、鳴かれていなければ −1 にする。イベントの番号は記録の中の位置
 * そのものなので、フィールドには書かない。ポンのイベントは、後で加槓してもポンの面子のまま書く。加槓は、O の副露のイベントとして
 * 加槓の面子を書く。鳴かれた打牌と副露が合っているか（鳴きは元の打牌の直後か、加槓は自分のポンからか）は、エンジンが記録を書く
 * ときに確かめるので、ここでは確かめない。北抜きは種類と席だけを書く。書く先は 0 で埋めてある前提で、イベントごとに {@code long}
 * に詰めてから丸ごと書く。メモリは確保しない。
 */
final class EventRecords {

  private EventRecords() {}

  /**
   * エンジンの記録にある全てのイベントを、記録 O のイベントのセクションに書く。
   *
   * @param schema 記録 O の形
   * @param log エンジンが記録した公開のイベント
   * @param observer 観測席（絶対席）
   * @param words 書く先の配列（イベントのセクションは 0 で埋めてあること）
   * @param at 書く先の配列で、イベントのセクションが始まる添字
   */
  static void write(ObservationSchema schema, EventLog log, int observer, long[] words, int at) {
    int seats = schema.variant().seats();
    int discard = schema.eventKinds.index(ObservationSchema.EventKind.DISCARD);
    int meld = schema.eventKinds.index(ObservationSchema.EventKind.MELD);
    int nuki = schema.eventKinds.contains(ObservationSchema.EventKind.NUKI)
        ? schema.eventKinds.index(ObservationSchema.EventKind.NUKI)
        : -1;
    for (int event = 0, events = log.size(); event < events; event++) {
      int author = Math.floorMod(log.seat(event) - observer, seats);
      EventKind kind = log.kind(event);
      long record =
          switch (kind) {
            case DISCARD -> {
              int calledBy = log.calledBy(event);
              long word = head(schema, discard, author);
              word = schema.discard.with(word, DiscardBits.TILE, log.tile(event));
              word = schema.discard.with(word, DiscardBits.AKA, log.red(event) ? 1 : 0);
              word = schema.discard.with(word, DiscardBits.TSUMOGIRI, log.tsumogiri(event) ? 1 : 0);
              word = schema.discard.with(word, DiscardBits.RIICHI, log.riichi(event) ? 1 : 0);
              yield schema.discard.with(
                  word,
                  DiscardBits.CALLED_BY,
                  calledBy == EventLog.NONE ? EventLog.NONE : Math.floorMod(calledBy - observer, seats));
            }
            case MELD, KAKAN -> schema.meld.with(head(schema, meld, author), MeldBits.MELD, log.meld(event));
            case NUKI -> head(schema, nuki, author);
          };
      schema.event.putRecord(words, at, event, record);
    }
  }

  /** イベントの記録のうち、どの種類にも共通する先頭部分（種類と、イベントを起こした席）を詰めた {@code long}。 */
  private static long head(ObservationSchema schema, int kind, int author) {
    return schema.event.with(schema.event.with(0L, EventBits.KIND, kind), EventBits.AUTHOR, author);
  }
}
