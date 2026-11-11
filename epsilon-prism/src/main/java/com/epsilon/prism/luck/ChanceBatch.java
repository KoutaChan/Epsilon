package com.epsilon.prism.luck;

import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import java.util.Objects;

/**
 * 前の判断の時点から今までの、偶然のイベントの束を読む。束は槓ドラの表示（めくった順）が先で、ツモ（山・嶺上）は高々 1 つで最後に来る。
 * 並びが崩れていれば、エンジンの決まりか読む時点の取り方が誤っているので、例外にする。
 *
 * <p>経路ごとに 1 つ持ち、判断の時点ごとに 1 回読む。写しと分岐は親の読んだ位置から始め、主経路は局が替わったら位置を戻す。局末の
 * 裏ドラの表示牌は最後の判断の時点の後にめくるので、束には入らない。
 */
final class ChanceBatch {

  /** 次に読むイベントの番号。 */
  private int next;

  /** 束の最初のイベントの番号。 */
  private int first;

  private int reveals;
  private int draw = -1;

  /** 局の始めから読む。 */
  ChanceBatch() {}

  /**
   * 親の読んだ位置から読む（写しと分岐）。束は空から始める。
   *
   * @param parent 親の経路の束
   */
  ChanceBatch(ChanceBatch parent) {
    next = parent.next;
    first = next;
  }

  /** 次の局の始めに位置を戻し、束を空にする。 */
  void reset() {
    next = 0;
    first = 0;
    reveals = 0;
    draw = -1;
  }

  /**
   * 前に読んだ位置から記録の終わりまでを、次の束として読む。
   *
   * @param log 経路の局の偶然のイベントの記録
   * @throws IllegalStateException 記録が前に読んだ位置より短いとき、ツモの後に表示があるとき、ツモが 2 つあるとき、裏ドラの表示牌が
   *     あるとき
   */
  void read(ChanceLog log) {
    int size = log.size();
    if (size < next) {
      throw new IllegalStateException("The chance log has " + size + " events but " + next + " were read");
    }
    first = next;
    reveals = 0;
    draw = -1;
    for (int index = first; index < size; index++) {
      ChanceKind kind = log.kind(index);
      switch (kind) {
        case INDICATOR -> {
          if (draw >= 0) {
            throw new IllegalStateException("Indicator " + index + " follows the draw " + draw + " in one batch");
          }
          reveals++;
        }
        case DRAW, RINSHAN -> {
          if (draw >= 0) {
            throw new IllegalStateException("Draws " + draw + " and " + index + " fall in one batch");
          }
          draw = index;
        }
        case URA -> throw new IllegalStateException("Ura indicator " + index + " before a decision");
      }
    }
    next = size;
  }

  /** 次に読むイベントの番号（最後に読んだ束の終わり）。 */
  int position() {
    return next;
  }

  /** 束のイベントの数。 */
  int size() {
    return reveals + (draw >= 0 ? 1 : 0);
  }

  /** 束の槓ドラの表示の数。 */
  int reveals() {
    return reveals;
  }

  /**
   * 束の中の表示牌の番号を返す。
   *
   * @param k 束の中で何番目の表示か（0 始まり、めくった順）
   * @return 記録の中の番号
   */
  int reveal(int k) {
    return first + Objects.checkIndex(k, reveals);
  }

  /** 束のツモの番号。ツモが無ければ −1。 */
  int draw() {
    return draw;
  }
}
