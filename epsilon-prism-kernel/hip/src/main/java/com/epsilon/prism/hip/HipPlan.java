package com.epsilon.prism.hip;

import ai.djl.Device;
import ai.djl.pytorch.engine.PtStream;
import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * 1 つのスロットの HIP の実行計画。作業領域（{@link HipRegion}）と重みのバッファは作るときに一度だけ確保し、処理は全てスロットの
 * 計算ストリームに積む。{@link #forward} の中では、同期もデバイスのメモリの確保もしない。
 *
 * <p>ストリームは DJL の {@link PtStream} で、計画より長く生かす。スロットのストリームのスコープで {@code PtEvent.record()} を呼べば、
 * 同じストリームにイベントが記録される。そのため、パイプラインで共通に使っているイベントでの同期が、そのまま使える。スレッドセーフでは
 * ない（1 つのスロットの submit のスレッドだけが使う）。
 */
public final class HipPlan implements AutoCloseable {

  private final HipShape shape;

  public HipShape shape() {
    return shape;
  }

  private long handle;
  private long weightsVersion = -1;

  private HipPlan(HipShape shape, long handle) {
    this.shape = shape;
    this.handle = handle;
  }

  /**
   * 計画を作る。GPU の arch を照合し、作業領域と重みのバッファを確保する。
   *
   * @param library 読み込み済みの libprism_hip.so
   * @param stream スロットの計算ストリーム（GPU）
   * @param shape 形と容量
   * @return 計画
   */
  public static HipPlan create(HipLibrary library, PtStream stream, HipShape shape) {
    return create(library, stream, shape, HipPlanOptions.BASE);
  }

  /** 配列の容量と追加の候補重み・マスクを同時に指定して計画を作る。 */
  public static HipPlan create(
      HipLibrary library, PtStream stream, HipShape shape, HipPlanOptions options) {
    Objects.requireNonNull(library, "library");
    Objects.requireNonNull(shape, "shape");
    Device device = stream.getDevice();
    if (!device.isGpu()) {
      throw new IllegalArgumentException("HIP plans need a GPU stream, got " + device);
    }
    HipShape.Capacity capacity = shape.capacity();
    long handle =
        PrismHip.newPlan(
            device.getDeviceId(),
            stream.getId(),
            shape.architecture().encode(),
            options.encode(),
            capacity.rows(),
            capacity.events(),
            capacity.transitions(),
            capacity.actions(),
            capacity.inputBytes(),
            shape.weightBytes());
    return new HipPlan(shape, handle);
  }

  /** 公開イベントを使う転送で、候補の追加の区分を復元する列と混合基数を設定する。 */
  public void configureCandidateInput(int[] descriptor) {
    PrismHip.configureCandidateInput(handle(), descriptor);
  }

  /** VRAM の確保前後の空きと総量（バイト）、確保できたイベント数。 */
  public record EventMemory(long freeBefore, long freeAfter, long total, int slots) {}

  /** 残りのレーン数で空き VRAM を分け、このレーンの保存領域を確保する。イベント数 0 は自動。 */
  public EventMemory configureEventPages(int remaining, int events, double reserveFraction) {
    long[] values = PrismHip.configureEventPages(handle(), remaining, events, reserveFraction);
    return new EventMemory(values[0], values[1], values[2], Math.toIntExact(values[3]));
  }

  /** 最後に写した WeightPack のバージョン（まだ写していなければ −1）。 */
  public long weightsVersion() {
    return weightsVersion;
  }

  /**
   * WeightPack をデバイスへ写す。写すのは position から limit までで、先頭にヘッダーを含む。native はヘッダーを読み、形の記述が計画の
   * ものとビット単位で同じこと、全てのバッファが WeightPack の中にあることを照合する。その後、重みから作る表（関係のバイアスの表と、
   * 重みをフラグメントの順に並べ替えた写し）を作り直す。公開のときだけ呼び、ストリームを同期する。
   *
   * @param pack WeightPack（direct な ByteBuffer）
   * @param version ネットワークのバージョン（0 以上）
   */
  public void uploadWeights(ByteBuffer pack, long version) {
    requireDirect(pack);
    if (version < 0) {
      throw new IllegalArgumentException("weights version must not be negative, got " + version);
    }
    if (pack.remaining() > shape.weightBytes() || pack.remaining() < HipPack.HEADER_BYTES) {
      throw new IllegalArgumentException(
          "weight pack of "
              + pack.remaining()
              + " bytes does not fit the plan capacity "
              + shape.weightBytes());
    }
    PrismHip.uploadWeights(handle(), pack, pack.position(), pack.remaining(), version);
    weightsVersion = version;
  }

  /**
   * 1 バッチの順伝播をストリームに積む。blob の H2D を 1 回 → 形の記述の順にカーネル（{@link HipKernel}）→ s の D2H を 1 回。
   * 同期はしない（終わりはストリームのイベントで待つ）。
   *
   * @param input blob（position から {@link HipBatch#bytes()} バイト。direct な ByteBuffer で、推論ではページ固定）
   * @param batch バッチの中の数（行・イベント・遷移・合法手）と、blob の中の領域の位置
   * @param scores s を書く先（position から {@link HipBatch#actions()} 個の FP32。要求の順、little-endian）
   * @param version 期待する重みのバージョン（写したバージョンと違えば例外）
   */
  public void forward(ByteBuffer input, HipBatch batch, ByteBuffer scores, long version) {
    forward(input, batch, scores, version, null, 0, 0);
  }

  /** 配置表には、行ごとのクエリタイルのマスク、イベントの保存先、今回計算するイベントの番号を順に並べる。 */
  public void forward(
      ByteBuffer input,
      HipBatch batch,
      ByteBuffer scores,
      long version,
      ByteBuffer eventMap,
      int misses,
      int slots) {
    forward(input, batch, scores, version, eventMap, misses, slots, 0);
  }

  /** 未収載のイベントの入力を配置表の後ろから送る。転送バイト数が 0 の場合は全入力を使う。 */
  public void forward(
      ByteBuffer input,
      HipBatch batch,
      ByteBuffer scores,
      long version,
      ByteBuffer eventMap,
      int misses,
      int slots,
      int transferBytes) {
    if (transferBytes < 0
        || transferBytes > 0 && (eventMap == null || eventMap.capacity() < transferBytes)) {
      throw new IllegalArgumentException("invalid event payload length " + transferBytes);
    }
    requireDirect(input);
    requireDirect(scores);
    shape.requireFits(batch);
    if (version != weightsVersion) {
      throw new IllegalStateException(
          "weight pack version "
              + weightsVersion
              + " is stale; the network is at version "
              + version);
    }
    if (input.remaining() < batch.bytes()) {
      throw new IllegalArgumentException(
          "input holds " + input.remaining() + " bytes, batch needs " + batch.bytes());
    }
    long scoreBytes = (long) batch.actions() * Float.BYTES;
    if (scores.remaining() < scoreBytes) {
      throw new IllegalArgumentException(
          "scores hold " + scores.remaining() + " bytes, batch needs " + scoreBytes);
    }
    if (eventMap == null) {
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
    } else {
      PrismHip.forwardPaged(
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
          version,
          eventMap,
          misses,
          slots,
          transferBytes);
    }
  }

  /**
   * 作業領域の今の中身のまま、1 つのカーネルを回す。カーネルごとの一致のテストと計測だけで使う。入力の blob は、先に {@link #copyIn}
   * で {@link HipRegion#INPUT} に置いておく。
   *
   * @param kernel カーネル
   * @param layer 層（{@link HipKernel#acceptsLayer} が受ける層）
   * @param batch バッチの中の数（行・イベント・遷移・合法手）と、blob の中の領域の位置
   */
  public void run(HipKernel kernel, int layer, HipBatch batch) {
    shape.requireFits(batch);
    if (!kernel.acceptsLayer(layer, shape.architecture())) {
      throw new IllegalArgumentException(
          kernel + " does not run layer " + layer + " of " + shape.architecture());
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

  /** 作業領域の中の指定した領域の先頭へ、バッファの position から指定のバイト数を写す。ストリームに積むだけで、終わりは待たない。 */
  public void copyIn(HipRegion region, ByteBuffer source, long bytes) {
    requireRange(region, source, bytes);
    PrismHip.copyIn(handle(), region.ordinal(), source, source.position(), bytes);
  }

  /** 作業領域の中の指定した領域の先頭から指定のバイト数を、バッファの position へ写す。写した中身は {@link #synchronize()} の後に読む。 */
  public void copyOut(HipRegion region, ByteBuffer target, long bytes) {
    requireRange(region, target, bytes);
    PrismHip.copyOut(handle(), region.ordinal(), target, target.position(), bytes);
  }

  /** カーネルの計時を入れるか切るかを決める。入れると、カーネルの前後にイベントを記録する。テスト、計測、反復ごとの報告で使う。 */
  public void profile(boolean enabled) {
    PrismHip.profile(handle(), enabled);
  }

  /**
   * 最後の forward か run での、カーネルごとの時間（ms、{@link HipKernel} の順）。層ごとに回すカーネルは全ての層の和、回していない
   * カーネルは NaN。終わりのイベントを待ってから返す。
   */
  public float[] kernelMillis() {
    return PrismHip.kernelMillis(handle());
  }

  /** ストリームに積んだ処理が終わるのを待つ。テストと計測だけで使う。 */
  public void synchronize() {
    PrismHip.synchronize(handle());
  }

  /** ストリームに積んだ処理が終わるのを待ってから、作業領域と重みのバッファを解放する。 */
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
