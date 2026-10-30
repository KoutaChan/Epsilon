package com.epsilon.prism.policy;

import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.FrontierFeatures;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputWriter;
import com.epsilon.prism.input.Tokenizer;
import com.epsilon.prism.observe.Observation;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * 1 回の受け付けの問い合わせを網の入力にした作業域（{@link BatchEncoder} の池で使い回す）。
 *
 * <p>入力は問い合わせが運ぶ記録 O だけ（{@link PolicyRequest#capture}。出す側が自分のスレッドで捕捉し、答えを書き終えるまで書き換えない）。符号化の
 * スレッドは公開の観測にも局面にも触れない。O の読み手は行ごとに 1 つ持ち、受け付けごとに 1 回だけ付ける（節の位置と合法手を読んだまま、計画と
 * 符号化で使う）。読み手は池に返しても残し、次の受け付けで付け替える。
 *
 * <ol>
 *   <li>付け（{@link #bind}）: 行ごとに読み手を問い合わせの O に付ける。範囲ごとに 1 つのスレッドから呼ぶ。
 *   <li>計画（{@link #plan}）: 1 回の順伝播の詰めたトークン（行の数 × 最も長い行の S）が上限に収まれば、問い合わせの順のまま 1 つの塊にする。
 *       収まらなければ S の順に並べ、上限と塊の容量に収まるだけ詰めた塊に分ける（行は切らない）。
 *   <li>符号化（{@link #tokenize}）: {@link Tokenizer} が塊の行の位置へ直に書く。
 *   <li>答え（{@link #answer}）: 塊の順に塊の手の順の s をつないだ出力から、問い合わせごとに s・p0・σ を書く（{@link GreedyAnswer}）。
 * </ol>
 *
 * <p>計画の位置 p は塊の順に続き、塊 c は位置 {@link Chunk#first} から塊の行の数だけを持つ。位置 p の問い合わせは {@code planned[p]}。
 */
final class Encoding {

  /**
   * 1 回の順伝播の塊。
   *
   * @param batch 計画を封じた入力
   * @param first 塊の最初の行の計画の位置
   */
  record Chunk(HostInputs batch, int first) {}

  private final List<Chunk> chunks = new ArrayList<>();
  private List<PolicyRequest> rows;

  /** 行ごとの O の読み手（{@link #bind} で付け、{@link #rows} の行の数だけ使う）。 */
  private Observation[] observations = new Observation[0];

  private int[] planned = new int[0];
  private long[] keys = new long[0];
  private long paddedTokens;

  /**
   * 問い合わせの並びに付け替える（池から借りたときに 1 回）。
   *
   * @param requests 問い合わせ（1 つ以上。結果を書き終えるまで借りる）
   * @throws IllegalArgumentException 問い合わせがないとき
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
        observations[row] = new Observation();
      }
      planned = new int[count];
      keys = new long[count];
    }
  }

  /** 問い合わせと塊を手放す（池に返す前）。 */
  void clear() {
    rows = null;
    chunks.clear();
  }

  /** 問い合わせの並び。 */
  List<PolicyRequest> rows() {
    return rows;
  }

  /** 塊（計画の順）。 */
  List<Chunk> chunks() {
    return chunks;
  }

  /** 全ての塊の詰めたトークンの和（デバイスの仕事の見積もり）。 */
  long paddedTokens() {
    return paddedTokens;
  }

  /** 全ての塊の手の数（出力の s の数）。 */
  int actions() {
    int actions = 0;
    for (Chunk chunk : chunks) {
      actions += chunk.batch().plan().count(Axis.ACTION);
    }
    return actions;
  }

  /**
   * 範囲の問い合わせの読み手を、問い合わせが運ぶ O に付ける。範囲ごとに 1 つのスレッドから呼ぶ。
   *
   * @param from 最初の問い合わせ
   * @param to 最後の問い合わせの次
   * @throws IllegalStateException O を運ばない問い合わせがあるとき
   */
  void bind(int from, int to) {
    for (int row = from; row < to; row++) {
      PolicyRequest request = rows.get(row);
      observations[row].bind(request.recordWords(), request.recordBase());
    }
  }

  /**
   * 塊を計画して封じる（付けの後に 1 つのスレッドで）。
   *
   * @param batches 空の塊を借りる口
   * @param budget 1 回の順伝播の詰めたトークンの上限
   */
  void plan(Supplier<HostInputs> batches, int budget) {
    int count = rows.size();
    int widest = 0;
    for (int row = 0; row < count; row++) {
      int tokens = BatchPlan.FIXED_TOKENS + observations[row].publicEvents();
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
      // 並べた後の行の S は塊の中で最も長いので、詰めたトークンは（行の数）× S
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

  /** 問い合わせの順のまま塊に入れる（塊の容量を超えたら false）。 */
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
   * 計画の位置の範囲の行を塊に書く。範囲ごとに 1 つのスレッドから呼ぶ。
   *
   * @param from 最初の位置
   * @param to 最後の位置の次
   * @param tokenizer このスレッドの Tokenizer
   * @param writer このスレッドの書き手
   * @param frontier このスレッドの前線の書き手（前線のない網は null）
   */
  void tokenize(
      int from, int to, Tokenizer tokenizer, InputWriter writer, FrontierFeatures frontier) {
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
      tokenizer.write(o, writer.bind(target.batch().input(), row));
      if (frontier != null) {
        frontier.write(o, target.batch().frontier(), row);
      }
    }
  }

  /**
   * 問い合わせごとに s・p0・σ を書く。
   *
   * @param output 塊の順に、塊の手の順の s を詰めてつないだもの（位置 0 から読む。HIP の推論の出力と同じ並び）
   * @param settings p0 の係数
   */
  void answer(FloatBuffer output, PriorSettings settings) {
    int offset = 0;
    for (Chunk target : chunks) {
      BatchPlan plan = target.batch().plan();
      for (int row = 0; row < plan.rows(); row++) {
        PolicyRequest request = rows.get(planned[target.first() + row]);
        int actions = request.legal().size();
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

  /** p0（特徴は Tokenizer が作った打牌の後の向聴と受け入れの枚数）。 */
  private static float[] prior(
      PolicyRequest request, HostBatch batch, int firstAction, PriorSettings settings) {
    int actions = request.legal().size();
    float[] shanten = new float[actions];
    float[] ukeire = new float[actions];
    for (int action = 0; action < actions; action++) {
      shanten[action] = batch.priorShanten(firstAction + action);
      ukeire[action] = batch.priorUkeire(firstAction + action);
    }
    return Prior.compute(request.legal(), shanten, ukeire, settings);
  }
}
