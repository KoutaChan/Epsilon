package com.epsilon.prism.hip;

import ai.djl.Device;
import ai.djl.pytorch.engine.PtStream;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * 1 つの slot の HIP の実行計画。作業域（{@link HipRegion}）と重みの置き場は作るときに一度だけ取り、仕事は全て slot の計算ストリームに
 * 積む。{@link #forward} の中では、同期もデバイスのメモリの確保もしない。
 *
 * <p>ストリームは DJL の {@link PtStream} で、計画より長く生かす。{@code PtEvent.record()} を slot のストリームのスコープで呼べば、
 * 同じストリームに印が載るので、共通の pipeline の待ち合わせがそのまま使える。スレッド安全ではない（1 つの slot の submit の
 * スレッドだけが使う）。
 */
public final class HipPlan implements AutoCloseable {

  private final HipShape shape;
  private final PtStream stream;
  private long handle;
  private long weightsVersion = -1;

  private HipPlan(HipShape shape, PtStream stream, long handle) {
    this.shape = shape;
    this.stream = stream;
    this.handle = handle;
  }

  /**
   * 計画を作る。GPU の arch を照らし、作業域と重みの置き場を取る。
   *
   * @param library 読み込み済みの libprism_hip.so
   * @param stream slot の計算ストリーム（GPU）
   * @param shape 形と容量
   * @return 計画
   */
  public static HipPlan create(HipLibrary library, PtStream stream, HipShape shape) {
    Objects.requireNonNull(library, "library");
    Objects.requireNonNull(shape, "shape");
    Device device = stream.getDevice();
    if (!device.isGpu()) {
      throw new IllegalArgumentException("HIP plans need a GPU stream, got " + device);
    }
    library.requireArchitecture(device.getDeviceId());
    HipShape.Capacity capacity = shape.capacity();
    long handle =
        PrismHip.newPlan(
            device.getDeviceId(),
            stream.getId(),
            shape.hidden(),
            shape.ffn(),
            shape.gate(),
            shape.depth(),
            shape.epsilon(),
            shape.fixedTokens(),
            capacity.rows(),
            capacity.events(),
            capacity.transitions(),
            capacity.actions(),
            capacity.inputBytes(),
            shape.weightBytes());
    return new HipPlan(shape, stream, handle);
  }

  public HipShape shape() {
    return shape;
  }

  /** 計画を作ったストリーム。 */
  public PtStream stream() {
    return stream;
  }

  /** 最後に写した WeightPack の版（まだなら −1）。 */
  public long weightsVersion() {
    return weightsVersion;
  }

  /**
   * WeightPack（{@code pack} の position から limit まで、頭つき）をデバイスへ写す。native が頭を読み、全ての置き場が内にあることを
   * 照らし、重みから作る表（K2・K4 の関係の偏りの表）を作り直す。公開のときだけ呼び、ストリームを同期する。
   *
   * @param pack 直の ByteBuffer
   * @param version 網の版の番号（0 以上）
   */
  public void uploadWeights(ByteBuffer pack, long version) {
    requireDirect(pack);
    if (version < 0) {
      throw new IllegalArgumentException("weights version must not be negative, got " + version);
    }
    if (pack.remaining() > shape.weightBytes() || pack.remaining() < HipPack.HEADER_BYTES) {
      throw new IllegalArgumentException(
          "weight pack of " + pack.remaining() + " bytes does not fit the plan capacity " + shape.weightBytes());
    }
    PrismHip.uploadWeights(handle(), pack, pack.position(), pack.remaining(), version);
    weightsVersion = version;
  }

  /**
   * 1 塊の順伝播をストリームに積む: blob の H2D 1 回 → K1〜K7 → s の D2H 1 回。同期はしない（終わりはストリームの印で待つ）。
   *
   * @param input blob（position から {@link HipBatch#bytes()} バイト、直の ByteBuffer。推論ではページ固定）
   * @param batch 塊の数と区画の位置
   * @param scores s を書く先（position から {@link HipBatch#actions()} 個の FP32、要求の順、little-endian）
   * @param version 期待する重みの版（写した版と違えば例外）
   */
  public void forward(ByteBuffer input, HipBatch batch, ByteBuffer scores, long version) {
    requireDirect(input);
    requireDirect(scores);
    shape.requireFits(batch);
    if (version != weightsVersion) {
      throw new IllegalStateException(
          "weight pack version " + weightsVersion + " is stale; the network is at version " + version);
    }
    if (input.remaining() < batch.bytes()) {
      throw new IllegalArgumentException("input holds " + input.remaining() + " bytes, batch needs " + batch.bytes());
    }
    long scoreBytes = (long) batch.actions() * Float.BYTES;
    if (scores.remaining() < scoreBytes) {
      throw new IllegalArgumentException("scores hold " + scores.remaining() + " bytes, batch needs " + scoreBytes);
    }
    PrismHip.forward(
        handle(),
        input,
        input.position(),
        batch.rows(),
        batch.events(),
        batch.transitions(),
        batch.actions(),
        batch.bytes(),
        batch.sections(),
        scores,
        scores.position(),
        version);
  }

  /**
   * 作業域の今の中身のまま 1 つのカーネルを回す（カーネルごとの一致の試験と測りだけで使う）。入力の blob は先に {@link #copyIn} で
   * {@link HipRegion#INPUT} に置く。
   *
   * @param kernel カーネル
   * @param layer 層（K2・K3 は最後でない層、K4・K5 は最後の層、ほかは 0）
   * @param batch 塊の数と区画の位置
   */
  public void run(HipKernel kernel, int layer, HipBatch batch) {
    shape.requireFits(batch);
    if (!kernel.acceptsLayer(layer, shape.depth())) {
      throw new IllegalArgumentException(kernel + " does not run layer " + layer + " of depth " + shape.depth());
    }
    if (weightsVersion < 0) {
      throw new IllegalStateException("weights have not been uploaded to the plan");
    }
    PrismHip.run(
        handle(),
        kernel.ordinal(),
        layer,
        batch.rows(),
        batch.events(),
        batch.transitions(),
        batch.actions(),
        batch.bytes(),
        batch.sections());
  }

  /** 区画の先頭 {@code bytes} バイトへ、{@code source} の position から写す（ストリームに積むだけ）。 */
  public void copyIn(HipRegion region, ByteBuffer source, long bytes) {
    requireRange(region, source, bytes);
    PrismHip.copyIn(handle(), region.ordinal(), source, source.position(), bytes);
  }

  /** 区画の先頭 {@code bytes} バイトを、{@code target} の position へ写す。{@link #synchronize()} の後に読む。 */
  public void copyOut(HipRegion region, ByteBuffer target, long bytes) {
    requireRange(region, target, bytes);
    PrismHip.copyOut(handle(), region.ordinal(), target, target.position(), bytes);
  }

  /** カーネルの計時を入れる・切る（印をカーネルの前後に置く。試験・測り・反復ごとの報告で使う）。 */
  public void profile(boolean enabled) {
    PrismHip.profile(handle(), enabled);
  }

  /**
   * 最後の forward（または run）でのカーネルごとの時間（ms、{@link HipKernel} の順。層ごとに回すものは和、回していないものは NaN）。
   * 終わりの印を待つ。
   */
  public float[] kernelMillis() {
    return PrismHip.kernelMillis(handle());
  }

  /** ストリームの仕事が終わるのを待つ（試験と測りだけで使う）。 */
  public void synchronize() {
    PrismHip.synchronize(handle());
  }

  /** ストリームの仕事が終わるのを待ってから、作業域と重みの置き場を返す。 */
  @Override
  public void close() {
    if (handle != 0) {
      long released = handle;
      handle = 0;
      PrismHip.freePlan(released);
    }
  }

  private long handle() {
    if (handle == 0) {
      throw new IllegalStateException("the HIP plan is closed");
    }
    return handle;
  }

  private void requireRange(HipRegion region, ByteBuffer buffer, long bytes) {
    requireDirect(buffer);
    if (bytes < 0 || bytes > region.capacityBytes(shape)) {
      throw new IllegalArgumentException(
          region + " holds " + region.capacityBytes(shape) + " bytes, cannot copy " + bytes);
    }
    if (bytes > buffer.remaining()) {
      throw new IllegalArgumentException(
          region + " copy of " + bytes + " bytes exceeds the buffer's " + buffer.remaining());
    }
  }

  private static void requireDirect(ByteBuffer buffer) {
    if (!buffer.isDirect()) {
      throw new IllegalArgumentException("HIP transfers need a direct ByteBuffer");
    }
  }
}
