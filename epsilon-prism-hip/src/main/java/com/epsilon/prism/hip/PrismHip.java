package com.epsilon.prism.hip;

import java.nio.ByteBuffer;

/**
 * {@code libprism_hip.so} の JNI の口。網の名前は知らず、作業域の区画・カーネル・自分を説明する WeightPack（{@link HipPack}）・
 * 塊の数と区画の位置だけを受け取る。
 *
 * <p>{@link HipLibrary#load()} が読み込んだ後だけ呼べる。引数は {@link HipPlan} が先に確かめ、native も同じ上限を照らす。
 * ホストのメモリは直の {@link ByteBuffer}（推論ではページ固定の {@code PtPinnedBuffer}）、ストリームは {@code PtStream.getId()}
 * で受ける。
 */
final class PrismHip {

  private PrismHip() {}

  /** 読み込まれた HIP の実行系と libtorch の事実を {@code 名前=値} の列で返す（{@link HipLibrary.RuntimeInfo}）。 */
  static native String[] runtimeInfo();

  /** GPU {@code device} の arch（{@code gcnArchName}）。 */
  static native String deviceArch(int device);

  /** 計画を作り、作業域と重みの置き場を一度だけ取る。 */
  static native long newPlan(
      int device,
      long stream,
      int hidden,
      int ffn,
      int gate,
      int depth,
      float epsilon,
      int fixedTokens,
      int rows,
      int events,
      int transitions,
      int actions,
      long inputBytes,
      long weightBytes);

  static native void freePlan(long plan);

  /** WeightPack（頭つき）を読み、照らしてからデバイスへ写す。ストリームを同期する。 */
  static native void uploadWeights(long plan, ByteBuffer pack, int position, long bytes, long version);

  static native void copyIn(long plan, int region, ByteBuffer source, int position, long bytes);

  static native void copyOut(long plan, int region, ByteBuffer target, int position, long bytes);

  /** 作業域の入力のまま 1 つのカーネルを回す（試験と測りだけで使う）。 */
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

  /** H2D 1 回 → K1〜K7 → s の D2H 1 回をストリームに積む。 */
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

  /** DJL の配列（torch::Tensor* の手持ち）のデータのデバイスのアドレス。要素の型と GPU を照らす（{@link HipAttention}）。 */
  static native long tensorAddress(long handle, int element, int gpu);

  /** 学習の関係の符号を作る演算をストリームに積む（アドレスは {@link #tensorAddress}）。 */
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

  /** 学習の注意の順伝播をストリームに積む（アドレスは {@link #tensorAddress}）。 */
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

  /** 学習の注意の逆伝播をストリームに積む。 */
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

  /** 学習の RMSNorm の順伝播をストリームに積む（アドレスは {@link #tensorAddress}）。 */
  static native void normForward(
      long stream, int rows, int width, float epsilon, long input, long gain, long out, long inverseRms);

  /** 学習の RMSNorm の逆伝播をストリームに積む。 */
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
