package com.epsilon.prism.auxiliary;

import com.epsilon.mahjong.engine.EventKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;

/**
 * 経路ごと・席ごとに、最後に聴牌だったイベントの番号と、最後にロンの役がある聴牌だったイベントの番号を持つ。立直は役に数える。補助の
 * ヘッドが学ぶ「聴牌に届いたか」の教師に使う（{@link HandEndFacts#reachedTenpai}）。
 *
 * <p>イベントの番号はエンジンが公開するイベントの番号で、記録 O のイベントと同じ数え方。卓のコアが打牌を行うたびに {@link #discarded} を
 * 呼ぶと、その打牌の後の 13 枚を見て、聴牌ならその打牌の番号を書く。判断の時点までのイベントの数（記録 O のヘッダーにある値）を e とすると、
 * 番号が e 以上なら判断の後に聴牌に届いたことになる。聴牌と役の判定は {@link TenpaiProbe} に従い、役の有無はドラに依らない。
 *
 * <p>分岐は親の写し（{@link #TenpaiReach(TenpaiReach)}）から始める。局が替わると番号は 0 から数え直すので、局を続ける経路（主経路）は
 * 次の局の前に {@link #reset} を呼ぶ。
 */
public final class TenpaiReach {

  /** 番号がない（この局でまだその形になっていない）ことを表す値。 */
  public static final int NONE = -1;

  private final int[] tenpai;
  private final int[] yakuTenpai;

  /**
   * どの席も番号のない状態（局の始め）で作る。
   *
   * @param variant ルール
   */
  public TenpaiReach(Variant variant) {
    tenpai = new int[variant.seats()];
    yakuTenpai = new int[variant.seats()];
    reset();
  }

  /**
   * 写しを作る。分岐を始めるときと、局末の事実に残すときに使う。
   *
   * @param other 写す元
   */
  public TenpaiReach(TenpaiReach other) {
    tenpai = other.tenpai.clone();
    yakuTenpai = other.yakuTenpai.clone();
  }

  /** 全ての席の番号を消す。次の局の前に呼ぶ。 */
  public void reset() {
    Arrays.fill(tenpai, NONE);
    Arrays.fill(yakuTenpai, NONE);
  }

  /**
   * 打牌の後の局面を見て、打牌した席の手が聴牌ならその打牌の番号を書く。打牌の後、その席が次の牌を引く前に呼ぶ。その打牌で局が終わった
   * ときも、局末の事実を作る前に呼ぶ。
   *
   * @param full 打牌の後の局面
   * @param seat 打牌した席
   * @param event 打牌のイベントの番号
   * @throws IllegalStateException 手が 13 枚相当でないとき
   * @throws IllegalArgumentException 番号がその席の打牌のイベントでないとき
   */
  public void discarded(FullView full, int seat, int event) {
    EventLog events = full.seat(seat).events();
    if (event < 0
        || event >= events.size()
        || events.kind(event) != EventKind.DISCARD
        || events.seat(event) != seat) {
      throw new IllegalArgumentException("Event " + event + " is not a discard of seat " + seat);
    }
    TenpaiProbe probe = TenpaiProbe.of(full.variant());
    if (!probe.tenpai(full, seat)) {
      return;
    }
    tenpai[seat] = event;
    if (probe.yakuTenpai(full, seat)) {
      yakuTenpai[seat] = event;
    }
  }

  /**
   * 席が最後に聴牌だったイベントの番号を返す。
   *
   * @param seat 席
   * @return 番号。なければ {@link #NONE}
   */
  public int lastTenpaiEvent(int seat) {
    return tenpai[seat];
  }

  /**
   * 席が最後にロンの役がある聴牌だったイベントの番号を返す。
   *
   * @param seat 席
   * @return 番号。なければ {@link #NONE}
   */
  public int lastYakuTenpaiEvent(int seat) {
    return yakuTenpai[seat];
  }
}
