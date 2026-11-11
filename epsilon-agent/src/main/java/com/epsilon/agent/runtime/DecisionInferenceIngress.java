package com.epsilon.agent.runtime;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * 推論のバッチ 1 つを符号化するために予約した、ホスト側の容量。予約の返し方も持つ。
 *
 * <p>GPU のスロットは予約しない。符号化に失敗したら予約を返す（{@link #abort}）。所有権を移した後（{@link #handoff}）は、受け取った側が、
 * 予約を実行先へ渡す（{@link Ownership#move}）か返す（{@link Ownership#release}）かのどちらかを、一度だけ行う。
 */
public final class DecisionInferenceIngress {

  private final Object owner;
  private final Object route;
  private final Runnable release;
  private State state = State.OPEN;

  /**
   * @param owner 予約した評価器。所有権を移すときに、同じ評価器かを確かめる
   * @param route 実行先へ渡す予約（{@link Ownership#move} が返す）。容量を予約しない評価器では null
   * @param release 予約を返す処理
   */
  public DecisionInferenceIngress(Object owner, Object route, Runnable release) {
    this.owner = Objects.requireNonNull(owner, "owner");
    this.route = route;
    this.release = Objects.requireNonNull(release, "release");
  }

  /** 容量を数えない予約を作る。呼び出したスレッドで推論する評価器が使う。 */
  public static DecisionInferenceIngress direct(Object owner) {
    return new DecisionInferenceIngress(owner, null, () -> {});
  }

  /** 待たずに容量を取ろうとした結果。取れたときは予約を持つ。取れなかったときは、次に容量が空いたときに完了する通知を持つ。 */
  public record Attempt(DecisionInferenceIngress ingress, CompletableFuture<Void> available) {

    public Attempt {
      if ((ingress == null) == (available == null)) {
        throw new IllegalArgumentException("An ingress attempt holds exactly one of an ingress and a wake");
      }
    }

    public static Attempt acquired(DecisionInferenceIngress ingress) {
      return new Attempt(Objects.requireNonNull(ingress, "ingress"), null);
    }

    public static Attempt blocked(CompletableFuture<Void> available) {
      return new Attempt(null, Objects.requireNonNull(available, "available"));
    }

    public boolean acquired() {
      return ingress != null;
    }
  }

  /** 符号化したバッチを渡すために、予約の所有権を移す。移せるのは一度だけ。予約した評価器以外が呼ぶと IllegalArgumentException を投げる。 */
  public synchronized Ownership handoff(Object expectedOwner) {
    if (owner != expectedOwner) {
      throw new IllegalArgumentException("The ingress belongs to another evaluator");
    }
    require(State.OPEN);
    state = State.HANDED_OFF;
    return new Ownership(route, release);
  }

  /** 符号化に失敗したときに、まだ移していない予約を返す。 */
  public synchronized void abort() {
    require(State.OPEN);
    state = State.ABORTED;
    release.run();
  }

  /** 予約がまだ移されても返されてもいなければ、返す。所有権を移した後か、返した後なら何もしない。 */
  public synchronized void abortIfOpen() {
    if (state == State.OPEN) {
      abort();
    }
  }

  private void require(State expected) {
    if (state != expected) {
      throw new IllegalStateException("The ingress is " + state + ", expected " + expected);
    }
  }

  /** 所有権を移した後の予約。実行先へ渡すか返すかを、一度だけ行う。 */
  public static final class Ownership {

    private final Object route;
    private final Runnable release;
    private State state = State.OPEN;

    private Ownership(Object route, Runnable release) {
      this.route = route;
      this.release = release;
    }

    /**
     * 予約を実行先へ渡す。
     *
     * @return 渡す予約。容量を予約しない評価器では null
     */
    public synchronized Object move() {
      require(State.OPEN);
      state = State.MOVED;
      return route;
    }

    public synchronized void release() {
      require(State.OPEN);
      state = State.RELEASED;
      release.run();
    }

    private void require(State expected) {
      if (state != expected) {
        throw new IllegalStateException("The ingress ownership is " + state + ", expected " + expected);
      }
    }
  }

  private enum State {
    OPEN,
    HANDED_OFF,
    ABORTED,
    MOVED,
    RELEASED
  }
}
