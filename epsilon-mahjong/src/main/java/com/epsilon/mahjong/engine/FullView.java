package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;

/**
 * 伏せた牌も読めるビュー。全ての席の手牌と、運で決まったイベントを読める。牌山の並びと、まだめくっていない表示牌・裏ドラ表示牌は
 * 出さないので、まだ見えていない牌どうしは入れ替えても区別がつかないまま残る。局を次に進める呼び出しまで有効。
 */
public interface FullView {

  Variant variant();

  SeatView seat(int seat);

  ChanceLog chance();

  /** その種類で、誰にも見えない枚数。{@link ChanceLog#unseen} と同じ。 */
  int unseen(int kind);

  /**
   * その物理牌が誰にも見えないなら true。牌山のまだ読んでいない位置にある牌が、これにあたる。{@link ChanceLog#hidden} と同じ。
   *
   * @param physicalId 物理 id
   * @throws IllegalArgumentException ルールに無い物理 id のとき
   */
  boolean hidden(int physicalId);

  /** 形だけで見たその席の和了牌（牌の種類ごとのビット）。手牌が 13 枚相当のときだけ求め、14 枚の間は 0。 */
  long waits(int seat);

  /** その席が見逃しの振聴（同巡内か、立直の後）なら true。自分の河による振聴は含まない。 */
  boolean furiten(int seat);

  /**
   * その席の純手牌（種類ごとの枚数と形のコード）と面子の数をコピーする。メモリは確保しない。
   *
   * @param seat 席
   * @param out コピー先。前の中身は捨てる
   */
  void copyHand(int seat, HandShape out);

  /**
   * その打牌でロンできるなら、裏ドラを数えない点を書いて true を返す。
   *
   * @param seat ロンする席
   * @param tile 打たれた牌の物理 id
   * @param from 打った席
   * @param ws ルールの作業領域
   * @param out 点を書く先。false のときの中身に意味は無い
   * @return ロンできれば true
   */
  boolean ronScore(int seat, int tile, int from, RuleWorkspace ws, WinScore out);

  /**
   * 直前にツモった牌が、指定した種類の赤でない牌だったとして、ツモ和了の点を書く。点に裏ドラは数えない。和了の出どころ
   * （嶺上開花・海底・天和・地和）、一発、立直、風、表示牌、抜きは実際のツモと同じにする。実際に引いた牌が赤でも、その赤は数えない。
   * 直前のイベントがその席のツモでなければ IllegalStateException を投げる。ルールに無い種類、ツモの前の手牌・面子・抜きで既に
   * 4 枚ある種類、ルールの違う作業領域には IllegalArgumentException を投げる。
   *
   * @param seat 直前にツモった席
   * @param kind 引いたとする牌の種類
   * @param ws ルールの作業領域
   * @param out 点を書く先。false のときの中身に意味は無い
   * @return 和了の形で役があれば true
   */
  boolean tsumoScore(int seat, int kind, RuleWorkspace ws, WinScore out);

  /**
   * 直前にツモった牌が、指定した物理牌だったとして、ツモ和了の点を書く。{@link #tsumoScore} と同じだが、指定した牌が赤なら
   * その赤を数える。実際に引いた牌が赤で、指定した牌がそれと違えば、引いた赤は数えない。
   *
   * @param seat 直前にツモった席
   * @param physicalId 引いたとする牌の物理 id。誰にも見えない牌か、実際に引いた牌
   * @param ws ルールの作業領域
   * @param out 点を書く先。false のときの中身に意味は無い
   * @return 和了の形で役があれば true
   * @throws IllegalStateException 直前のイベントがその席のツモでないとき
   * @throws IllegalArgumentException ルールに無い物理 id、見えている牌（実際に引いた牌は除く）、ルールの違う作業領域のとき
   */
  boolean tsumoScoreTile(int seat, int physicalId, RuleWorkspace ws, WinScore out);
}
