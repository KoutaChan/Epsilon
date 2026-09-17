package com.epsilon.runtime;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.jni.JniUtils;
import java.nio.file.Path;
import java.util.Objects;

/** 通常の速度測定から分離した、一バッチ・一区間だけの同期診断。 */
public final class InferenceProfile {
  private static final ThreadLocal<InferenceProfile> ACTIVE = new ThreadLocal<>();

  private final String selectedSection;
  private final Path traceFile;
  private final Device device;
  private final boolean recordOperators;
  private boolean claimed;
  private Result result;

  /** 実行記録の親ディレクトリは診断の呼び出し元が用意する。 */
  public InferenceProfile(String section, Path traceFile, Device device, boolean recordOperators) {
    if (section.isBlank()) throw new IllegalArgumentException("Profile section must not be blank");
    selectedSection = section;
    this.traceFile = Objects.requireNonNull(traceFile, "traceFile");
    this.device = Objects.requireNonNull(device, "device");
    this.recordOperators = recordOperators;
  }

  /** 診断バッチを処理する間だけ、現在のスレッドに計測設定を関連付ける。 */
  public static Attachment attach(InferenceProfile profile) {
    if (profile == null) return null;
    InferenceProfile previous = ACTIVE.get();
    ACTIVE.set(profile);
    return new Attachment(previous);
  }

  /** 指定された区間の計測を一度だけ開始する。診断が無効な場合や対象外の区間ではnullを返す。 */
  public static Scope section(String name) {
    InferenceProfile profile = ACTIVE.get();
    if (profile == null || profile.claimed || !profile.selectedSection.equals(name)) return null;
    profile.claimed = true;
    return new Scope(profile);
  }

  /** 対象バッチの回収後に読む。選択区間を通らなければnull。 */
  public Result result() {
    return result;
  }

  /** 実経過時間はCPU発行と診断用の完了待ちを含む。ネイティブ側で取得した実行記録のGPU時間とは区別する。 */
  public record Result(
      String section,
      String device,
      long enqueueNanos,
      long completedWallNanos,
      String traceFile) {}

  public static final class Attachment implements AutoCloseable {
    private final InferenceProfile previous;

    private Attachment(InferenceProfile previous) {
      this.previous = previous;
    }

    @Override
    public void close() {
      if (previous == null) ACTIVE.remove();
      else ACTIVE.set(previous);
    }
  }

  public static final class Scope implements AutoCloseable {
    private final InferenceProfile profile;
    private final boolean nativeTrace;
    private final PtEvent completion;
    private final long startedNanos;

    private Scope(InferenceProfile profile) {
      this.profile = profile;
      boolean deviceSection = !profile.selectedSection.startsWith("host.");
      nativeTrace = deviceSection && profile.recordOperators;
      PtEvent event = null;
      try {
        if (deviceSection && profile.device.isGpu()) {
          event = ((PtEngine) Engine.getEngine("PyTorch")).newEvent(profile.device);
          event.record();
          event.synchronize();
        }
        if (nativeTrace) JniUtils.startProfile(profile.device.isGpu(), true, false);
      } catch (RuntimeException | Error failure) {
        if (event != null) {
          try {
            event.close();
          } catch (RuntimeException | Error closeFailure) {
            failure.addSuppressed(closeFailure);
          }
        }
        throw failure;
      }
      completion = event;
      startedNanos = System.nanoTime();
    }

    @Override
    public void close() {
      long enqueueNanos = System.nanoTime() - startedNanos;
      long completedWallNanos = enqueueNanos;
      Throwable failure = null;
      try {
        if (completion != null) {
          completion.record();
          completion.synchronize();
          completedWallNanos = System.nanoTime() - startedNanos;
        }
      } catch (RuntimeException | Error completionFailure) {
        failure = completionFailure;
      }
      if (nativeTrace) {
        try {
          JniUtils.stopProfile(profile.traceFile.toString());
        } catch (RuntimeException | Error stopFailure) {
          failure = appendFailure(failure, stopFailure);
        }
      }
      if (completion != null) {
        try {
          completion.close();
        } catch (RuntimeException | Error closeFailure) {
          failure = appendFailure(failure, closeFailure);
        }
      }
      if (failure instanceof RuntimeException runtime) throw runtime;
      if (failure instanceof Error error) throw error;
      profile.result =
          new Result(
              profile.selectedSection,
              profile.device.toString(),
              enqueueNanos,
              completedWallNanos,
              nativeTrace ? profile.traceFile.toString() : null);
    }
  }

  private static Throwable appendFailure(Throwable failure, Throwable next) {
    if (failure == null) return next;
    failure.addSuppressed(next);
    return failure;
  }
}
