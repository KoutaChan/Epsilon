package com.epsilon.prism.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputFormat;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * 1 回の受付の要求を、ネットワークの入力にするための作業領域。{@link BatchEncoder} のプールで使い回す。
 *
 * <p>入力は要求が運ぶ記録 O だけ。O は要求を出す側が自分のスレッドで取り込み、答えを書き終えるまで書き換えない。符号化のスレッドは、
 * 公開の観測にも局面にも触れない。O のリーダーは行ごとに 1 つ持ち、受付ごとに 1 回だけ O に付ける。付けたときに読んだセクションの位置と
 * 合法手は、計画と符号化でそのまま使う。リーダーはプールに返しても残し、次の受付で付け替える。
 *
 * <ol>
 *   <li>付ける（{@link #bind}）: 行ごとに、リーダーを要求の O に付ける。範囲ごとに 1 つのスレッドから呼ぶ。
 *   <li>計画（{@link #plan}）: 1 回の順伝播で扱うトークン数（パディング込みで、行の数 × 最も長い行の S）が上限に収まり、バッチの容量にも
 *       入れば、要求の順のまま 1 つのバッチにする。そうでなければ S の順に並べ、上限とバッチの容量に収まるだけ詰めたバッチに分ける
 *       （行は切らない）。
 *   <li>符号化（{@link #tokenize}）: 入力の形式の 1 行のライターが、バッチの行の位置へ直に書く。
 *   <li>答え（{@link #answer}）: s をバッチの順・手の順につないだ出力から、要求ごとに s・p0・σ を書く（{@link GreedyAnswer}）。
 * </ol>
 *
 * <p>計画の位置 p はバッチの順に続き、バッチ c は位置 {@link Chunk#first} からバッチの行の数だけを持つ。位置 p の要求は {@code planned[p]}。
 */
final class Encoding {

  /**
   * 1 回の順伝播で扱うバッチ。
   *
   * @param batch 計画を確定した入力
   * @param first バッチの最初の行の位置（計画の中の位置）
   */
  record Chunk(HostInputs batch, int first) {}

  private final InputFormat format;
  private final List<Chunk> chunks = new ArrayList<>();
  private List<PolicyRequest> rows;

  /** 行ごとの O のリーダー。{@link #bind} で付け、{@link #rows} の行の数だけ使う。 */
  private Observation[] observations = new Observation[0];

  private int[] planned = new int[0];
  private long[] keys = new long[0];
  private long paddedTokens;

  /**
   * 空の作業領域を作る。
   *
   * @param format 入力の形式。O のルール（四人打ちか三人打ち）と、行ごとの局・席・牌のトークンの数を持つ
   */
  Encoding(InputFormat format) {
    this.format = format;
  }

  /**
   * 新しい要求の並びに付け替える。プールから借りたときに 1 回呼ぶ。
   *
   * @param requests 方策の要求（1 つ以上。結果を書き終えるまで借りる）
   * @throws IllegalArgumentException 要求が無いとき
   */
  void reset(List<PolicyRequest> requests) {
    if (requests.isEmpty()) {
      throw new IllegalArgumentException("no policy requests");
    }
    rows = requests;
    chunks.clear();
    paddedTokens = 0;
    int count = requests.size();
    if (planned.length < count) {
      int had = observations.length;
      observations = Arrays.copyOf(observations, count);
      for (int row = had; row < count; row++) {
        observations[row] = new Observation(format.variant());
      }
      planned = new int[count];
      keys = new long[count];
    }
  }

  /** 要求とバッチへの参照を外す。プールに返す前に呼ぶ。 */
  void clear() {
    rows = null;
    chunks.clear();
  }

  /** 要求の並び。 */
  List<PolicyRequest> rows() {
    return rows;
  }

  /** バッチの並び（計画の順）。 */
  List<Chunk> chunks() {
    return chunks;
  }

  /** 全てのバッチで扱うトークン数（パディング込み）の合計。デバイスの処理量の見積もりに使う。 */
  long paddedTokens() {
    return paddedTokens;
  }

  /** 全てのバッチの手の数（出力の s の数）。 */
  int actions() {
    int actions = 0;
    for (Chunk chunk : chunks) {
      actions += chunk.batch().plan().count(Axis.ACTION);
    }
    return actions;
  }

  /**
   * 範囲の各行で、リーダーを要求が運ぶ O に付ける。範囲ごとに 1 つのスレッドから呼ぶ。
   *
   * @param from 最初の要求の位置
   * @param to 最後の要求の次の位置
   * @throws IllegalStateException O を運ばない要求があるとき
   */
  void bind(int from, int to) {
    for (int row = from; row < to; row++) {
      PolicyRequest request = rows.get(row);
      observations[row].bind(request.recordWords(), request.recordBase());
    }
  }

  /**
   * バッチを計画して確定する。リーダーを付けた後に、1 つのスレッドで呼ぶ。
   *
   * @param batches 空のバッチを借りる先
   * @param budget 1 回の順伝播で扱うトークン数（パディング込み）の上限
   */
  void plan(Supplier<HostInputs> batches, int budget) {
    int count = rows.size();
    int widest = 0;
    for (int row = 0; row < count; row++) {
      int tokens = format.fixedTokens() + observations[row].publicEvents();
      keys[row] = (long) tokens << Integer.SIZE | row;
      widest = Math.max(widest, tokens);
    }
    HostInputs spare = null;
    if ((long) count * widest <= budget) {
      spare = batches.get();
      if (fill(spare)) {
        seal(spare, 0, widest);
        return;
      }
      spare.plan().clear();
    }
    Arrays.sort(keys, 0, count);
    HostInputs batch = null;
    int first = 0;
    for (int position = 0; position < count; position++) {
      int row = (int) keys[position];
      int tokens = (int) (keys[position] >>> Integer.SIZE);
      Observation o = observations[row];
      // S の順に並べたので、この行の S がバッチの中で最も長い。パディング込みのトークン数は（行の数）× S
      if (batch == null
          || (long) (position - first + 1) * tokens > budget
          || batch.plan().add(o) < 0) {
        if (batch != null) {
          seal(batch, first, (int) (keys[position - 1] >>> Integer.SIZE));
        }
        batch = spare != null ? spare : batches.get();
        spare = null;
        first = position;
        if (batch.plan().add(o) < 0) {
          throw new IllegalStateException("a row does not fit an empty batch");
        }
      }
      planned[position] = row;
    }
    seal(batch, first, (int) (keys[count - 1] >>> Integer.SIZE));
  }

  /** 要求の順のままバッチに入れる。バッチの容量を超えたら false を返す。 */
  private boolean fill(HostInputs batch) {
    for (int row = 0; row < rows.size(); row++) {
      if (batch.plan().add(observations[row]) < 0) {
        return false;
      }
      planned[row] = row;
    }
    return true;
  }

  private void seal(HostInputs batch, int first, int widest) {
    batch.plan().seal();
    chunks.add(new Chunk(batch, first));
    paddedTokens += (long) batch.plan().rows() * widest;
  }

  /**
   * 計画の位置の範囲にある行を、バッチに書く。範囲ごとに 1 つのスレッドから呼ぶ。
   *
   * @param from 最初の位置
   * @param to 最後の位置の次
   * @param writer このスレッドの 1 行のライター
   */
  void tokenize(int from, int to, InputFormat.RowWriter writer) {
    int chunk = 0;
    while (chunk + 1 < chunks.size() && chunks.get(chunk + 1).first() <= from) {
      chunk++;
    }
    for (int position = from; position < to; position++) {
      while (chunk + 1 < chunks.size() && chunks.get(chunk + 1).first() <= position) {
        chunk++;
      }
      Chunk target = chunks.get(chunk);
      Observation o = observations[planned[position]];
      int row = position - target.first();
      writer.write(o, target.batch(), row);
    }
  }

  /**
   * 要求ごとに s・p0・σ を書く。
   *
   * @param output 合法手の s を、バッチの順、バッチの中では手の順に詰めてつないだもの（位置 0 から読む。HIP の推論の出力と同じ並び）
   * @param settings p0 の係数
   */
  void answer(FloatBuffer output, PriorSettings settings) {
    int offset = 0;
    for (Chunk target : chunks) {
      BatchPlan plan = target.batch().plan();
      for (int row = 0; row < plan.rows(); row++) {
        PolicyRequest request = rows.get(planned[target.first() + row]);
        int actions = request.legalCount();
        if (actions != plan.actions(row)) {
          throw new IllegalStateException(
              "row holds " + plan.actions(row) + " actions, the request " + actions);
        }
        float[] scores = new float[actions];
        output.get(offset + plan.firstAction(row), scores);
        GreedyAnswer.write(
            request,
            scores,
            GreedyAnswer.needsPrior(request, scores)
                ? prior(request, target.batch().input(), plan.firstAction(row), settings)
                : null);
      }
      offset += plan.count(Axis.ACTION);
    }
  }

  /** p0 を作る。特徴は、ライターが書いた打牌の後の向聴数と受け入れの枚数。 */
  private static float[] prior(
      PolicyRequest request, HostBatch batch, int firstAction, PriorSettings settings) {
    int actions = request.legalCount();
    float[] shanten = new float[actions];
    float[] ukeire = new float[actions];
    for (int action = 0; action < actions; action++) {
      shanten[action] = batch.priorShanten(firstAction + action);
      ukeire[action] = batch.priorUkeire(firstAction + action);
    }
    return Prior.compute(request.legal(), shanten, ukeire, settings);
  }
}
