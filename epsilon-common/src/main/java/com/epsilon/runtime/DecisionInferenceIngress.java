package com.epsilon.runtime;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * ホスト上で推論バッチを構築するために確保した容量と、その解放手順を管理する。
 *
 * <p>GPUの実行枠は確保しない。入力の符号化に失敗した場合は{@link #abort()}で容量を返す。 {@link
 * #handoff(Object)}で所有権を移した後は、受け取り側が実行先への移譲または解放を担当する。
 */
public final class DecisionInferenceIngress {

  private final Object owner;
  private final Object route;
  private final Runnable release;
  private State state = State.OPEN;

  public DecisionInferenceIngress(Object owner, Object route, Runnable release) {
    this.owner = Objects.requireNonNull(owner);
    this.route = route;
    this.release = Objects.requireNonNull(release);
  }

  public static DecisionInferenceIngress direct(Object owner) {
    return new DecisionInferenceIngress(owner, null, () -> {});
  }

  /** 待機せずに入力バッチ用の容量を確保しようとした結果。容量不足の場合は、次に容量が空くときの通知を返す。 */
  public record Attempt(DecisionInferenceIngress ingress, CompletableFuture<Void> available) {

    public Attempt {
      if ((ingress == null) == (available == null)) {
        throw new IllegalArgumentException(
            "Decision inference ingress attempt must contain exactly one result");
      }
    }

    public static Attempt acquired(DecisionInferenceIngress ingress) {
      return new Attempt(Objects.requireNonNull(ingress), null);
    }

    public static Attempt blocked(CompletableFuture<Void> available) {
      return new Attempt(null, Objects.requireNonNull(available));
    }

    /** 入力バッチの構築に必要な容量を確保できた場合だけtrueを返す。 */
    public boolean acquired() {
      return ingress != null;
    }
  }

  /** 符号化済みのバッチを引き渡すため、予約容量の所有権を一度だけ移す。 */
  public synchronized Ownership handoff(Object expectedOwner) {
    if (owner != expectedOwner) {
      throw new IllegalArgumentException("Decision inference ingress belongs to another evaluator");
    }
    require(State.OPEN);
    state = State.HANDED_OFF;
    return new Ownership(route, release);
  }

  /** 入力の符号化に失敗したとき、まだ移譲していない予約容量を解放する。 */
  public synchronized void abort() {
    require(State.OPEN);
    state = State.ABORTED;
    release.run();
  }

  private void require(State expected) {
    if (state != expected) {
      throw new IllegalStateException(
          "Decision inference ingress is " + state + ", expected " + expected);
    }
  }

  /** 移譲後の予約を管理する。実行先への引き渡しか解放のどちらかを一度だけ行う。 */
  public static final class Ownership {

    private final Object route;
    private final Runnable release;
    private State state = State.OPEN;

    private Ownership(Object route, Runnable release) {
      this.route = route;
      this.release = release;
    }

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
        throw new IllegalStateException(
            "Decision inference ingress ownership is " + state + ", expected " + expected);
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
