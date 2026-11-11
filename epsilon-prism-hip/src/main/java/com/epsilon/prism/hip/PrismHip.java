package com.epsilon.prism.hip;

import java.nio.ByteBuffer;

/**
 * {@code libprism_hip.so} の JNI のインターフェース。ネットワークの側の名前は知らない。受け取るのは、作業領域の各領域、カーネル、
 * 自分の形を書いた WeightPack（{@link HipPack}）、バッチの各数、入力の各領域の位置だけ。
 *
 * <p>{@link HipLibrary#load()} で読み込んだ後だけ呼べる。引数は {@link HipPlan} が先に確かめ、native も同じ上限で照合する。
 * ホストのメモリは direct の {@link ByteBuffer}（推論ではページ固定の {@code PtPinnedBuffer}）で、ストリームは
 * {@code PtStream.getId()} で受け取る。
 */
final class PrismHip {

  private PrismHip() {}

  /** 読み込んだ HIP のランタイムと libtorch の情報を、{@code 名前=値} の配列で返す（{@link HipLibrary.RuntimeInfo}）。 */
  static native String[] runtimeInfo();

  /** GPU {@code device} の arch（{@code gcnArchName}）。 */
  static native String deviceArch(int device);

  /** 計画を作り、作業領域と重みのバッファを一度だけ確保する（architecture は {@link HipArchitecture#encode()} の値）。 */
  static native long newPlan(
      int device,
      long stream,
      int[] architecture,
      int rows,
      int events,
      int transitions,
      int actions,
      long inputBytes,
      long weightBytes);

  static native void freePlan(long plan);

  /** ヘッダーつきの WeightPack を読み、照合してからデバイスへ写す。ストリームを同期する。 */
  static native void uploadWeights(long plan, ByteBuffer pack, int position, long bytes, long version);

  static native void copyIn(long plan, int region, ByteBuffer source, int position, long bytes);

  static native void copyOut(long plan, int region, ByteBuffer target, int position, long bytes);

  /** 作業領域にある入力のまま、カーネルを 1 つだけ実行する（テストと計測でだけ使う）。 */
  static native void run(
      long plan,
      int kernel,
      int layer,
      int rows,
      int events,
      int transitions,
      int actions,
      long bytes,
      long[] sections);

  /** 入力の H2D を 1 回、形の記述の順のカーネル、手のスコア s の D2H を 1 回、この順にストリームに積む。 */
  static native void forward(
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

  static native void profile(long plan, boolean enabled);

  static native float[] kernelMillis(long plan);

  static native void synchronize(long plan);

  /** DJL の配列（torch::Tensor* のハンドル）が持つデータの、デバイス上のアドレス。要素の型と GPU を照合する（{@link HipAttention}）。 */
  static native long tensorAddress(long handle, int element, int gpu);

  /** 学習で使う関係のコードを作る演算を、ストリームに積む（アドレスは {@link #tensorAddress} で得る）。 */
  static native void relationCodes(
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
      long stream, int rows, int width, float epsilon, long input, long gain, long out, long inverseRms);

  /** 学習の RMSNorm の逆伝播を、ストリームに積む。 */
  static native void normBackward(
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
