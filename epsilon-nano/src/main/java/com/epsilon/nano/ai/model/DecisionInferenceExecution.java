package com.epsilon.nano.ai.model;

import ai.djl.ndarray.NDManager;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonMahjongStateReadoutGroupExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonPlayerMemoryFusionExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonStrategicContextFusionExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonTileRelationFusionExecution;

/**
 * 一つのDecision デバイスパイプラインが所有する状態エンコーダー、特徴量の集約、方策の永続推論実行計画を束ねる。
 *
 * <p>同じバッチのプレイヤーごとの履歴表現、牌種間の関係、戦略文脈、特徴量の集約、方策出力は同時に開始し、最終D2Hが完了した時だけ一緒に解放する。
 * これにより、後段カーネルがまだ参照している循環バッファ枠を次バッチが上書きしない。
 */
public final class DecisionInferenceExecution implements AutoCloseable {

  private final EpsilonPlayerMemoryFusionExecution playerMemory;
  private final EpsilonTileRelationFusionExecution tileRelation;
  private final EpsilonStrategicContextFusionExecution strategic;
  private final EpsilonMahjongStateReadoutGroupExecution readout;
  private final DecisionPolicyExecution policy;
  private boolean closed;

  DecisionInferenceExecution(
      EpsilonPlayerMemoryFusionExecution playerMemory,
      EpsilonTileRelationFusionExecution tileRelation,
      EpsilonStrategicContextFusionExecution strategic,
      EpsilonMahjongStateReadoutGroupExecution readout,
      DecisionPolicyExecution policy) {
    this.playerMemory = playerMemory;
    this.tileRelation = tileRelation;
    this.strategic = strategic;
    this.readout = readout;
    this.policy = policy;
  }

  /** 空いている共通枠で一つのネットワーク順伝播を開始する。 */
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("decision inference execution is closed");
    }
    EpsilonPlayerMemoryFusionExecution.Forward playerMemoryForward = null;
    EpsilonTileRelationFusionExecution.Forward tileRelationForward = null;
    EpsilonStrategicContextFusionExecution.Forward strategicForward = null;
    EpsilonMahjongStateReadoutGroupExecution.Forward readoutForward = null;
    try {
      playerMemoryForward = playerMemory == null ? null : playerMemory.beginForward(workingManager);
      tileRelationForward = tileRelation == null ? null : tileRelation.beginForward(workingManager);
      strategicForward = strategic == null ? null : strategic.beginForward(workingManager);
      readoutForward = readout == null ? null : readout.beginForward(workingManager);
      return new Forward(
          policy.beginForward(workingManager),
          playerMemoryForward,
          tileRelationForward,
          strategicForward,
          readoutForward);
    } catch (RuntimeException | Error failure) {
      if (readoutForward != null) {
        readoutForward.poison(failure);
      }
      if (strategicForward != null) {
        strategicForward.poison(failure);
      }
      if (tileRelationForward != null) {
        tileRelationForward.poison(failure);
      }
      if (playerMemoryForward != null) {
        playerMemoryForward.poison(failure);
      }
      throw failure;
    }
  }

  /** 完了未確認のデバイス処理を保持しているなら{@code true}を返す。 */
  public boolean hasIncompleteWork() {
    return policy.hasIncompleteWork()
        || (readout != null && readout.hasIncompleteWork())
        || (strategic != null && strategic.hasIncompleteWork())
        || (tileRelation != null && tileRelation.hasIncompleteWork())
        || (playerMemory != null && playerMemory.hasIncompleteWork());
  }

  /** 実行コンテキストを利用不能にした最初の失敗を返す。 */
  public Throwable failure() {
    Throwable policyFailure = policy.failure();
    if (policyFailure != null) {
      return policyFailure;
    }
    Throwable readoutFailure = readout == null ? null : readout.failure();
    if (readoutFailure != null) {
      return readoutFailure;
    }
    Throwable strategicFailure = strategic == null ? null : strategic.failure();
    if (strategicFailure != null) {
      return strategicFailure;
    }
    Throwable tileRelationFailure = tileRelation == null ? null : tileRelation.failure();
    return tileRelationFailure != null
        ? tileRelationFailure
        : playerMemory == null ? null : playerMemory.failure();
  }

  /** 同じストリーム末尾の完了確認後、失敗順伝播が保持した循環バッファ枠を解放する。 */
  public void releaseFailedForwardAfterCompletion() {
    Throwable failure = null;
    try {
      policy.releaseFailedForwardAfterCompletion();
    } catch (Throwable closeFailure) {
      failure = closeFailure;
    }
    if (readout != null) {
      try {
        readout.releaseFailedForwardAfterCompletion();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (strategic != null) {
      try {
        strategic.releaseFailedForwardAfterCompletion();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (tileRelation != null) {
      try {
        tileRelation.releaseFailedForwardAfterCompletion();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (playerMemory != null) {
      try {
        playerMemory.releaseFailedForwardAfterCompletion();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    rethrow(failure);
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    Throwable failure = null;
    try {
      policy.close();
    } catch (Throwable closeFailure) {
      failure = closeFailure;
    }
    if (readout != null) {
      try {
        readout.close();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (strategic != null) {
      try {
        strategic.close();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (tileRelation != null) {
      try {
        tileRelation.close();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (playerMemory != null) {
      try {
        playerMemory.close();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (failure == null) {
      closed = true;
    }
    rethrow(failure);
  }

  /** 一つのネットワーク順伝播が保持する状態エンコーダーと方策の実行枠。 */
  public static final class Forward {

    private final DecisionPolicyExecution.Forward policy;
    private final EpsilonPlayerMemoryFusionExecution.Forward playerMemory;
    private final EpsilonTileRelationFusionExecution.Forward tileRelation;
    private final EpsilonStrategicContextFusionExecution.Forward strategic;
    private final EpsilonMahjongStateReadoutGroupExecution.Forward readout;
    private boolean sealed;
    private boolean poisoned;

    private Forward(
        DecisionPolicyExecution.Forward policy,
        EpsilonPlayerMemoryFusionExecution.Forward playerMemory,
        EpsilonTileRelationFusionExecution.Forward tileRelation,
        EpsilonStrategicContextFusionExecution.Forward strategic,
        EpsilonMahjongStateReadoutGroupExecution.Forward readout) {
      this.policy = policy;
      this.playerMemory = playerMemory;
      this.tileRelation = tileRelation;
      this.strategic = strategic;
      this.readout = readout;
    }

    DecisionPolicyExecution.Forward policy() {
      return policy;
    }

    EpsilonPlayerMemoryFusionExecution.Forward playerMemory() {
      return playerMemory;
    }

    EpsilonTileRelationFusionExecution.Forward tileRelation() {
      return tileRelation;
    }

    EpsilonStrategicContextFusionExecution.Forward strategic() {
      return strategic;
    }

    EpsilonMahjongStateReadoutGroupExecution.Forward readout() {
      return readout;
    }

    /** 全循環バッファ枠を最終D2H完了まで保持する依存オブジェクトへ変換する。 */
    public AutoCloseable seal() {
      if (sealed || poisoned) {
        throw new IllegalStateException("decision inference forward cannot be sealed twice");
      }
      Dependencies dependencies = new Dependencies();
      try {
        if (playerMemory != null) {
          dependencies.add(playerMemory.seal());
        }
        if (tileRelation != null) {
          dependencies.add(tileRelation.seal());
        }
        if (strategic != null) {
          dependencies.add(strategic.seal());
        }
        if (readout != null) {
          dependencies.add(readout.seal());
        }
        dependencies.add(policy.seal());
        sealed = true;
        return dependencies;
      } catch (RuntimeException | Error failure) {
        poison(failure);
        throw failure;
      }
    }

    /** 最終出力まで到達しなかった失敗を両実行計画へ伝播する。 */
    public void poison(Throwable failure) {
      if (poisoned) {
        return;
      }
      poisoned = true;
      policy.poison(failure);
      if (readout != null) {
        readout.poison(failure);
      }
      if (strategic != null) {
        strategic.poison(failure);
      }
      if (tileRelation != null) {
        tileRelation.poison(failure);
      }
      if (playerMemory != null) {
        playerMemory.poison(failure);
      }
    }
  }

  private static final class Dependencies implements AutoCloseable {

    private final AutoCloseable[] values = new AutoCloseable[5];
    private int size;

    private void add(AutoCloseable value) {
      values[size++] = value;
    }

    @Override
    public void close() {
      Throwable failure = null;
      for (int index = size - 1; index >= 0; index--) {
        try {
          values[index].close();
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
      rethrow(failure);
    }
  }

  private static Throwable addFailure(Throwable failure, Throwable additional) {
    if (failure == null) {
      return additional;
    }
    if (failure != additional) {
      failure.addSuppressed(additional);
    }
    return failure;
  }

  private static void rethrow(Throwable failure) {
    if (failure == null) {
      return;
    }
    if (failure instanceof RuntimeException runtimeException) {
      throw runtimeException;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    throw new IllegalStateException(failure);
  }
}
