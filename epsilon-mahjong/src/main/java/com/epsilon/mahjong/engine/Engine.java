package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;

/**
 * 半荘を進めるエンジン（天鳳のルール。四人打ちと三人打ち）。判断の時点・局末・終局でだけ止まる。そこで読む判断・選んだ手の結果・
 * 局末・ビューは、次に進める呼び出しまで有効。配列はルールで決まる大きさで一度だけ確保し、半荘を始める・配る・選んだ手で進める呼び出しは
 * メモリを確保しない。1 つのスレッドからだけ使う。
 *
 * <p>乱数で決まるのは局ごとの牌山だけで、牌山は種と epoch の純関数なので、種と選んだ手が同じなら結果はビット単位で同じになる。
 * ルールの作業領域は持たず、進める呼び出しの引数で受け取る。リスナーは持たず、記録したい側が判断の時点ごとに読む。
 */
public final class Engine {

  /** 半荘の最初の局で牌山を作る epoch。 */
  private static final int FIRST_EPOCH = 1;

  private enum Stage {
    /** 半荘も局も始めていない。 */
    IDLE,
    /** 次の {@link #advance} が局を配る。 */
    DEAL,
    DECISIONS,
    ROUND_END,
    GAME_END
  }

  private final Variant variant;
  private final Transitions round;

  /** 半荘の始めの卓。{@link #startGame} は、これをいまの局の卓へコピーするだけで、メモリを確保しない。 */
  private final TableState first;

  /** いまの局の始まりの卓。 */
  private final TableState table;

  private Stage stage = Stage.IDLE;

  /** いまの局を牌譜の牌山で配ったなら true。そのときは次の局の牌山が無い。 */
  private boolean recorded;

  public Engine(Variant variant) {
    this.variant = variant;
    round = new Transitions(variant);
    first = new TableState(variant);
    table = new TableState(variant);
  }

  /**
   * 半荘を東 1 局（全員が配給点）から始める。最初の局は {@link #advance} で配る。
   *
   * @param seed 牌山の種
   * @param drawOrder 半荘の全ての局のツモの順
   */
  public void startGame(long seed, DrawOrder drawOrder) {
    table.copyFrom(first);
    begin(seed, FIRST_EPOCH, drawOrder);
  }

  /**
   * 与えた局の始まりの卓から半荘を始める。最初の局の牌山は種と epoch で作り、局ごとに epoch が 1 進む。
   *
   * @param start 局の始まりの卓
   * @param seed 牌山の種
   * @param epoch 最初の局の牌山の epoch
   * @param drawOrder 半荘の全ての局のツモの順
   */
  public void startAt(RoundStart start, long seed, int epoch, DrawOrder drawOrder) {
    table.set(start);
    begin(seed, epoch, drawOrder);
  }

  /**
   * 与えた卓の 1 局を、牌譜の牌山で始める。席別ツモ列は使わない。局末の後に {@link #advance} を呼ぶと IllegalStateException を
   * 投げる。
   *
   * @param start 局の始まりの卓
   * @param wall 位置ごとの物理 id を並べた牌山。見えない位置は −1 で、その位置を読むと IllegalStateException を投げる
   */
  public void startRecordedRound(RoundStart start, int[] wall) {
    table.set(start);
    round.load(wall);
    recorded = true;
    stage = Stage.DEAL;
  }

  /** 半荘の始めか局末の後に、次の局を配って最初の判断の時点まで進める。 */
  public Boundary advance(RuleWorkspace ws) {
    requireWorkspace(ws);
    if (stage == Stage.ROUND_END && !recorded) {
      table.copyFrom(round.end().next());
    } else if (stage != Stage.DEAL) {
      throw new IllegalStateException("Nothing to deal at " + stage + (recorded ? " of a recorded round" : ""));
    }
    return arrive(round.deal(table, !recorded, ws));
  }

  /**
   * 判断する席ごとに選んだ手で答え、次の判断の時点か局末まで進める。
   *
   * @param choices 席ごとに選んだ手（合法手の何番目か）。先頭から判断する席の数だけを読む
   * @param ws ルールの作業領域
   * @return 止まった時点
   */
  public Boundary commit(int[] choices, RuleWorkspace ws) {
    return commit(choices, TurnEvent.NONE, ws);
  }

  /**
   * 判断する席ごとに選んだ手で答え、次の判断の時点か局末まで進める。手番の打牌・立直の打牌・北抜きでは、手から出す物理牌を指定する。
   * 牌譜の再生が、記録と同じ物理牌を手から出すために使う（同じ種類の牌のどれを打ったかは、手の指定からは決まらない）。
   *
   * @param choices 席ごとに選んだ手（合法手の何番目か）。先頭から判断する席の数だけを読む
   * @param tile 手番の席が打つか抜く物理 id。選んだ手と種類・赤かどうかが合い、手にあり、ツモ切りの手ならツモった牌、手出しならほかの
   *     牌であること。エンジンに選ばせるなら {@link TurnEvent#NONE}
   * @param ws ルールの作業領域
   * @return 止まった時点
   * @throws IllegalArgumentException 指定した牌が選んだ手に合わないとき、または牌を出さない手か他家の応答に牌を指定したとき
   */
  public Boundary commit(int[] choices, int tile, RuleWorkspace ws) {
    requireStage(Stage.DECISIONS);
    requireWorkspace(ws);
    Decisions decisions = round.decisions();
    if (choices.length < decisions.size()) {
      throw new IllegalArgumentException(decisions.size() + " decisions need choices, got " + choices.length);
    }
    for (int slot = 0; slot < decisions.size(); slot++) {
      if (choices[slot] < 0 || choices[slot] >= decisions.legalCount(slot)) {
        throw new IllegalArgumentException(
            "Choice " + choices[slot] + " is outside the " + decisions.legalCount(slot) + " legal actions of slot "
                + slot);
      }
    }
    return arrive(round.commit(choices, tile, ws));
  }

  /** 判断の時点で、判断する席ごとの合法手。判断の時点でだけ読め、次に進める呼び出しまで有効。 */
  public Decisions decisions() {
    requireStage(Stage.DECISIONS);
    return round.decisions();
  }

  /**
   * 直前に選んだ手で進めたとき、席ごとに選んだ手が行われたかと、自分だけ選び替えると結果が変わりえたか。次に進める呼び出しまで
   * 有効。
   */
  public Resolution resolution() {
    return round.resolution();
  }

  /** 局末。局末か終局で止まっている間だけ読める。 */
  public RoundEnd roundEnd() {
    if (stage != Stage.ROUND_END && stage != Stage.GAME_END) {
      throw new IllegalStateException("No round has ended at " + stage);
    }
    return round.end();
  }

  /** 指定した席から見た卓。次に進める呼び出しまで有効。 */
  public SeatView view(int seat) {
    return round.full().seat(seat);
  }

  /** 伏せた牌・牌山・運で決まったイベントも読めるビュー。次に進める呼び出しまで有効。 */
  public FullView full() {
    return round.full();
  }

  /**
   * いまの局の始まりの卓。局を配った後（判断の時点・局末・終局）でだけ読める。次に進める呼び出しまで有効で、呼ぶ側は書き換えない。
   *
   * @throws IllegalStateException 局を配る前のとき
   */
  public TableState handStart() {
    if (stage == Stage.IDLE || stage == Stage.DEAL) {
      throw new IllegalStateException("No round is dealt at " + stage);
    }
    return table;
  }

  /**
   * まだ読まれていない牌（山の読み位置から先、めくっていない表示牌と裏ドラ表示牌、まだ引いていない嶺上と抜きの補充）を一様に並べ直す。
   * 種から物理 id ごとに乱数のキーを作り、牌をキーの順に、まだ読まれていない位置へ位置の番号の順に置く。席別ツモ列のまだ使っていない
   * 項目も同じ置換で書き換えるので、列の並びはそのままで中身だけが入れ替わる。見えない牌の種類ごとの枚数と、読んだ位置の牌は変わらない。
   * 立直した席が和了を合法手に持っていれば、その和了の点に入る裏ドラを、新しい牌山で数え直す。メモリは確保しない。
   *
   * @param seed 並べ直しの種
   * @param ws ルールの作業領域
   * @throws IllegalStateException 判断の時点でないとき、牌譜の牌山の局のとき
   */
  public void reshuffleUnseen(long seed, RuleWorkspace ws) {
    requireStage(Stage.DECISIONS);
    requireWorkspace(ws);
    if (recorded) {
      throw new IllegalStateException("A recorded round has no unseen wall to reshuffle");
    }
    round.reshuffleUnseen(seed, ws);
  }

  /**
   * 判断の時点でだけ呼べる。牌山・席別ツモ列・判断・判断 ID・epoch を含む全てのフィールドをコピーした新しいエンジンを返す。コピーを進めても
   * 元は変わらない。
   */
  public Engine copy() {
    requireStage(Stage.DECISIONS);
    Engine copy = new Engine(variant);
    copy.round.copyFrom(round);
    copy.table.copyFrom(table);
    copy.stage = stage;
    copy.recorded = recorded;
    return copy;
  }

  private void begin(long seed, int epoch, DrawOrder drawOrder) {
    round.start(seed, epoch, drawOrder);
    recorded = false;
    stage = Stage.DEAL;
  }

  /** 局の進行が止まった時点へ移る。局末なら局の結果を精算して次の卓を書き、局末か終局で止まる。 */
  private Boundary arrive(Boundary reached) {
    if (reached == Boundary.DECISIONS) {
      stage = Stage.DECISIONS;
      return reached;
    }
    RoundEnd end = round.end();
    Settlement.apply(variant, table, end.outcome(), end.delta());
    Progression.next(table, end.outcome(), end.delta(), end.next());
    boolean over = end.next().gameOver();
    stage = over ? Stage.GAME_END : Stage.ROUND_END;
    return over ? Boundary.GAME_END : Boundary.ROUND_END;
  }

  private void requireStage(Stage expected) {
    if (stage != expected) {
      throw new IllegalStateException("The engine is at " + stage + ", not " + expected);
    }
  }

  private void requireWorkspace(RuleWorkspace ws) {
    if (ws.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + ws.variant());
    }
  }
}
