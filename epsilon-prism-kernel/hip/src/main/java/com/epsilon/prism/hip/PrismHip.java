package com.epsilon.prism.hip;

import com.epsilon.prism.kernel.CollectionLane;
import com.epsilon.prism.kernel.PackedWeightLayout;
import java.nio.ByteBuffer;

/**
 * {@code libprism_hip.so} の JNI のインターフェース。ネットワークの側の名前は知らない。受け取るのは、モデルの宣言、自分の形を書いた
 * WeightPack（{@link PackedWeightLayout}）、バッチの各数、入力の各領域の位置だけ。
 *
 * <p>{@link HipLibrary#load()} で読み込んだ後だけ呼べる。収集の引数は {@link CollectionLane} が先に確かめ、native も同じ上限で照合する。
 * ホストのメモリは direct の {@link ByteBuffer}（推論ではページ固定の {@code PtPinnedBuffer}）で、ストリームは
 * {@code PtStream.getId()} で受け取る。
 */
final class PrismHip extends CollectionLane.Bindings {
  static final PrismHip COLLECTION = new PrismHip();

  private PrismHip() {}

  /** 読み込んだ HIP のランタイムと libtorch の情報を、{@code 名前=値} の配列で返す（{@link HipLibrary.RuntimeInfo}）。 */
  static native String[] runtimeInfo();

  /** GPU {@code device} の arch（{@code gcnArchName}）。 */
  static native String deviceArch(int device);

  @Override
  protected native long declaredPlan(
      int device,
      long stream,
      ByteBuffer declaration,
      int length,
      int abi,
      int countOffset,
      int segmentsOffset,
      int rows,
      int events,
      int transitions,
      int actions,
      long inputBytes,
      long weightBytes);

  @Override
  protected native void freePlan(long plan);

  static native void countForward(
      int device,
      long stream,
      int rows,
      int transitions,
      int tiles,
      int codes,
      int width,
      long projected,
      long gates,
      long rowIndex,
      long ids,
      long output);

  static native void countBackward(
      int device,
      long stream,
      int rows,
      int transitions,
      int tiles,
      int codes,
      int width,
      long projected,
      long gates,
      long rowIndex,
      long ids,
      long gradOutput,
      long gradProjected,
      long gradGate);

  static native void segmentAttentionForward(
      int device,
      long stream,
      int[] dimensions,
      float scale,
      long[] inputs,
      long output,
      long probabilities);

  static native void segmentAttentionBackward(
      int device, long stream, int[] dimensions, float scale, long[] inputs, long[] gradients);

  /** ヘッダーつきの WeightPack を読み、照合してからデバイスへ写す。ストリームを同期する。 */
  @Override
  protected native void uploadWeights(
      long plan, ByteBuffer pack, int position, long bytes, long version);

  @Override
  protected native long publishedVersion(long plan);

  /** 入力の H2D を 1 回、形の記述の順のカーネル、手のスコア s の D2H を 1 回、この順にストリームに積む。 */
  @Override
  protected native void forward(
      long plan,
      ByteBuffer input,
      int inputPosition,
      int rows,
      int events,
      int transitions,
      int actions,
      long bytes,
      long[] sections,
      ByteBuffer scores,
      int scoresPosition,
      long version);

  @Override
  protected native void forwardPaged(
      long plan,
      ByteBuffer input,
      int inputPosition,
      int rows,
      int events,
      int transitions,
      int actions,
      long bytes,
      long[] sections,
      ByteBuffer scores,
      int scoresPosition,
      long version,
      ByteBuffer eventMap,
      int misses,
      int slots,
      int transferBytes);

  @Override
  protected native void configureCandidateInput(long handle, int[] descriptor);

  @Override
  protected native long[] configureEventPages(
      long handle, int remaining, int events, double reserveFraction);

  /** DJL の配列（torch::Tensor* のハンドル）が持つデータの、デバイス上のアドレス。要素の型と GPU を照合する（{@link HipAttention}）。 */
  static native long tensorAddress(long handle, int element, int gpu);

  /** 学習で使う関係のコードを作る演算を、ストリームに積む（アドレスは {@link #tensorAddress} で得る）。 */
  static native void relationCodes(
      int device,
      long stream,
      int rows,
      int events,
      int tokens,
      int seats,
      int tiles,
      int rowWidth,
      int rowFirstEvent,
      int rowEvents,
      int width,
      int tile,
      int owner,
      int event,
      int riichi,
      int lastDiscard,
      int link,
      int safety,
      int safeties,
      long relation,
      long rowColumn,
      long tileRelation,
      long codes);

  /** 学習の Attention の順伝播を、ストリームに積む（アドレスは {@link #tensorAddress} で得る）。 */
  static native void attentionForward(
      int device,
      long stream,
      int rows,
      int queries,
      int keys,
      int codeQueries,
      int heads,
      int words,
      int segments,
      int entries,
      float scale,
      int causalFixedTokens,
      long query,
      long keyValue,
      long mask,
      long codes,
      long table,
      long out,
      long outFloat,
      long logSumExp);

  /** 学習の Attention の逆伝播を、ストリームに積む。 */
  static native void attentionBackward(
      int device,
      long stream,
      int rows,
      int queries,
      int keys,
      int codeQueries,
      int heads,
      int words,
      int segments,
      int entries,
      float scale,
      int causalFixedTokens,
      long query,
      long keyValue,
      long mask,
      long codes,
      long table,
      long outFloat,
      long logSumExp,
      long gradOut,
      long gradQuery,
      long gradKeyValue,
      long gradTable);

  /** 学習の RMSNorm の順伝播を、ストリームに積む（アドレスは {@link #tensorAddress} で得る）。 */
  static native void normForward(
      int device,
      long stream,
      int rows,
      int width,
      float epsilon,
      long input,
      long gain,
      long out,
      long inverseRms);

  /** 学習の RMSNorm の逆伝播を、ストリームに積む。 */
  static native void normBackward(
      int device,
      long stream,
      int rows,
      int width,
      float epsilon,
      long input,
      long gain,
      long inverseRms,
      long gradOut,
      long gradInput,
      long gradGain);
}
