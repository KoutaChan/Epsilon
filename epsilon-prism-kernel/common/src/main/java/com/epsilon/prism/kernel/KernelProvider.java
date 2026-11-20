package com.epsilon.prism.kernel;

import com.epsilon.prism.kernel.ModelSchema.Model;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.Set;

/** 能力の照会では GPU のランタイムをロードしない。計画ごとにデバイスと提出先ストリームを固定する。 */
public interface KernelProvider {
  enum Backend {
    HIP,
    CUDA
  }

  enum Use {
    COLLECTION,
    TRAINING
  }

  enum Operation {
    RELATION_CODES,
    TOKEN_EMBEDDING,
    RELATION_ATTENTION,
    FEED_FORWARD,
    COUNT_TILES,
    SEGMENT_BIAS,
    CANDIDATE_ATTENTION,
    CANDIDATE_FFN,
    EVENT_REREAD,
    DISCARD_VALUE,
    COMPOSE_VALUES,
    FULL_SUBMISSION,
    EVENT_PAGES,
    RELATION_ATTENTION_BACKWARD,
    RMS_NORM,
    RMS_NORM_BACKWARD,
    COUNT_TILES_BACKWARD,
    SEGMENT_ATTENTION_BACKWARD
  }

  enum Cache {
    OFF,
    CAUSAL
  }

  record Capacity(int rows, int events, int transitions, int actions, int inputBytes) {
    public Capacity {
      DeclarationIO.positive(rows);
      DeclarationIO.size(events);
      DeclarationIO.positive(transitions);
      DeclarationIO.positive(actions);
      DeclarationIO.positive(inputBytes);
    }
  }

  record Request(
      Model model,
      Backend backend,
      int device,
      ModelSchema.Element precision,
      Use use,
      Capacity capacity,
      Cache cache) {
    public Request {
      ModelDeclarations.validate(model);
      Objects.requireNonNull(backend);
      DeclarationIO.size(device);
      Objects.requireNonNull(precision);
      Objects.requireNonNull(use);
      Objects.requireNonNull(capacity);
      Objects.requireNonNull(cache);
    }
  }

  /** このモデルとデバイスで実装している演算、および未対応の条件。 */
  record Capabilities(Set<Operation> operations, String detail) {
    public Capabilities {
      operations = Set.copyOf(operations);
      Objects.requireNonNull(detail);
    }
  }

  /**
   * 入力の各領域は宣言順。スコアは行ごとの元の合法手順に書き、完了まで両方のバッファを保持する。
   *
   * @param sections 入力の各領域の先頭のバイト位置（宣言の順）。バッチが持つので、呼び出し側は書き換えない
   */
  record Batch(
      int rows,
      int events,
      int transitions,
      int actions,
      long[] sections,
      ByteBuffer input,
      ByteBuffer scores) {
    public Batch {
      if (rows < 1
          || events < 0
          || transitions < rows
          || actions < rows
          || !input.isDirect()
          || !scores.isDirect()
          || scores.isReadOnly()
          || scores.remaining() < Math.multiplyExact(actions, Float.BYTES))
        throw new IllegalArgumentException("Invalid batch counts or buffers");
      input = input.slice().asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
      scores = scores.slice().order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public ByteBuffer input() {
      return input.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public ByteBuffer scores() {
      return scores.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }
  }

  /** 重みの公開と解放は提出済み処理の完了を保証する。提出は指定ストリームに読み出しまでを積む。 */
  interface Plan extends AutoCloseable {
    /**
     * このレーンの重みと派生表を指定された世代に切り替える。同じ計画に公開し直すので、デバイスのメモリは増えない。失敗した公開の後は、
     * 次の公開が済むまでどの世代も提出に使えない。
     *
     * @throws IllegalArgumentException 宣言・計画を作った時の候補の重み・pack の大きさが合わないか、世代が公開済みの世代より新しくないとき
     */
    void publish(PackedWeights weights);

    /** 容量と公開済みの世代を照合し、入力の転送からスコアの書き戻しまでを計画のストリームに積む。容量の検査は native が行う。 */
    void submit(Batch batch, long generation);

    @Override
    void close();
  }

  /** 行ごとの配置表と未収載イベントの入力を、レーンのスロット数と一緒に渡す。 */
  record Pages(ByteBuffer data, int misses, int slots) {
    public Pages {
      if (!data.isDirect() || misses < 0 || slots <= 0)
        throw new IllegalArgumentException("Invalid event page transfer");
      data = data.slice().asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public ByteBuffer data() {
      return data.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }
  }

  /**
   * レーンのイベントのページを確保した結果と、確保の前後のデバイスのメモリ（バイト）、計画が使う候補の層の部品。
   *
   * @param freeBefore 確保の前の空き
   * @param freeAfter 確保の後の空き
   * @param total デバイスのメモリの総量
   * @param slots イベントのスロットの数（0 ならページを使わない）
   * @param projectionRows 射影表の行の数
   * @param batchBytes 1 回の提出が使う配置表と作業の領域
   * @param slotBytes スロットに保存するイベントの入力・キー・バリューの領域
   * @param projectionBytes 射影表の領域
   * @param dictionary 頻出の候補のコードの辞書を使うか
   * @param splitCandidate 候補の Attention を前段と後段に分けて実行するか
   */
  record LaneCapacity(
      long freeBefore,
      long freeAfter,
      long total,
      int slots,
      int projectionRows,
      long batchBytes,
      long slotBytes,
      long projectionBytes,
      boolean dictionary,
      boolean splitCandidate) {
    /** native の計画が返す値の並び（上の順、真偽は 0 か 1）から作る。並びはここでだけ読む。 */
    public static LaneCapacity of(long[] values) {
      if (values.length != 10)
        throw new IllegalStateException(
            "Expected 10 lane capacity values from the collection plan, got " + values.length);
      return new LaneCapacity(
          values[0],
          values[1],
          values[2],
          Math.toIntExact(values[3]),
          Math.toIntExact(values[4]),
          values[5],
          values[6],
          values[7],
          flag(values[8]),
          flag(values[9]));
    }

    private static boolean flag(long value) {
      if (value != 0 && value != 1)
        throw new IllegalStateException("Expected a 0 or 1 lane capacity flag, got " + value);
      return value == 1;
    }
  }

  interface PagedPlan extends Plan {
    /** 空きメモリからこのレーンのイベント領域を確保する。 */
    LaneCapacity configurePages(int remainingLanes, int eventsPerLane, double reserveFraction);

    void submit(Batch batch, long generation, Pages pages);
  }

  Backend backend();

  int abiVersion();

  /** 指定された宣言・デバイス・容量で使える演算と、実行を制限する条件を返す。 */
  Capabilities capabilities(Request request);

  /** デバイスとストリームを照合し、重みを公開する前の計画を開く。 */
  Plan open(Request request, long stream);
}
