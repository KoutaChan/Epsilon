package com.epsilon.prism.input;

import com.epsilon.agent.observe.EventRecords.Prefix;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * 1 バッチ分のネットワークの入力。UINT8 の blob 1 つに、{@link InputBlob.Tensor} の順に各テンソルの領域を {@link BatchPlan} が決めた位置に
 * 置く（リトルエンディアン）。通常推論は blob を一度に送る。因果推論で公開順を共有するときは、確定したイベントの列を
 * 観測席の符号化済みの配列から直接読む。
 *
 * <p>blob は外から渡す {@link ByteBuffer}（ダイレクトバッファか、ページ固定のメモリ）。符号化のスレッドは行ごとに別の位置へ絶対位置で
 * 書くので、複数のスレッドが同じバッチの別の行を同時に書いてよい（入力のライターは、中身を共有する blob の複製をスレッドごとに持つ）。
 *
 * <p>blob のほかに、デバイスへは送らないホストだけの配列として、合法手ごとのルールベースの事前分布 p0 の特徴（打牌の後の向聴と受け入れ。
 * {@link PriorFeatures}）を持つ。
 */
public final class HostBatch {

  private final ByteBuffer blob;
  private final BatchPlan plan;
  private final float[] priorShanten;
  private final float[] priorUkeire;

  /** 行ごとの、前の判断までに符号化を終えたイベントの数（{@link #eventPrefix}）。 */
  private final int[] eventPrefix;

  private Prefix[] published;
  private ByteBuffer basicCodes;
  private EventHistory.Bytes[] eventRows;

  /** 公開順のイベントを区間から直接読むバッチを準備する。行を並列に書き始める前に呼ぶ。 */
  public void enablePublishedEvents() {
    if (!plan.format().eventsSettleOnPublication())
      throw new IllegalArgumentException("unstable public event columns");
    published = new Prefix[eventPrefix.length];
    eventRows = new EventHistory.Bytes[eventPrefix.length];
  }

  /** この行に対応する観測席の公開順を付ける。単独で取り込んだ O には null を渡す。 */
  public void publishedPrefix(int row, Prefix prefix) {
    if (published[row] == prefix) return;
    if (prefix != null) prefix.retain();
    if (published[row] != null) published[row].release();
    published[row] = prefix;
  }

  /** 受付が所有しているこの行の公開順。 */
  public Prefix publication(int row) {
    return published == null ? null : published[row];
  }

  /** 候補の基本符号を遷移順に続けて書けるページ固定領域を関連付ける。 */
  public void basicCodes(ByteBuffer target) {
    basicCodes = target;
  }

  void writeBasicCodes(int first, byte[] codes, int length) {
    if (basicCodes != null) basicCodes.put(first * plan.format().fixedTokens(), codes, 0, length);
  }

  void bindEvents(int row, EventHistory.Bytes bytes) {
    if (eventRows != null) eventRows[row] = bytes;
  }

  /** バッチの利用が終わった区間と符号化済みの列を返す。 */
  public void releaseEvents() {
    if (published != null) {
      for (int row = 0; row < plan.rows(); row++) publishedPrefix(row, null);
      Arrays.fill(eventRows, 0, plan.rows(), null);
    }
  }

  /** イベント以外の入力を置く通常のバッファ。差分推論のイベントは {@link #copyEvents} から読む。 */
  public ByteBuffer storage() {
    return blob;
  }

  /** 1 行の公開イベントの指定区間を、転送先の各列へ詰める。GPU から押し出された区間も同じ符号から取り出す。 */
  public void copyEvents(
      int row, int from, int count, ByteBuffer out, int categories, int scalars, int relations) {
    int c = plan.format().width(Tensor.EVENT_CATEGORY);
    int s = Short.BYTES * plan.format().width(Tensor.EVENT_SCALAR);
    int r = plan.format().width(Tensor.RELATION);
    EventHistory.Bytes encoded = eventRows == null ? null : eventRows[row];
    int past = encoded == null ? 0 : Math.clamp(eventPrefix[row] - from, 0, count);
    if (past > 0) encoded.copy(from, past, out, categories, scalars, relations);
    if (past == count) return;
    int first = plan.firstEvent(row) + from + past;
    int cat = plan.offset(Tensor.EVENT_CATEGORY) + first * c;
    int scalar = plan.offset(Tensor.EVENT_SCALAR) + first * s;
    int rel = plan.offset(Tensor.RELATION) + (plan.tokenStart(Axis.EVENT) + first) * r;
    if (out == blob
        && categories + past * c == cat
        && scalars + past * s == scalar
        && relations + past * r == rel) return;
    int fresh = count - past;
    out.put(categories + past * c, blob, cat, fresh * c);
    out.put(scalars + past * s, blob, scalar, fresh * s);
    out.put(relations + past * r, blob, rel, fresh * r);
  }

  /**
   * バッチを作る。
   *
   * @param format 入力の形式
   * @param capacity 1 つのバッチの上限
   * @param blob 入力を書くバッファ（{@link BatchPlan#maxBytes} 以上。このバッチが持ち、バイト順をリトルエンディアンにする）
   * @throws IllegalArgumentException バッファが足りないとき
   */
  public HostBatch(InputFormat format, BatchPlan.Capacity capacity, ByteBuffer blob) {
    plan = new BatchPlan(format, capacity);
    int required = BatchPlan.maxBytes(format, capacity);
    if (blob.capacity() < required) {
      throw new IllegalArgumentException(
          "blob holds " + blob.capacity() + " bytes but the capacity needs " + required);
    }
    this.blob = blob.order(ByteOrder.LITTLE_ENDIAN);
    priorShanten = new float[capacity.actions()];
    priorUkeire = new float[capacity.actions()];
    eventPrefix = new int[capacity.rows()];
  }

  /**
   * 行の位置と各テンソルの領域の位置を決める計画を返す。
   *
   * @return 計画
   */
  public BatchPlan plan() {
    return plan;
  }

  /**
   * 全ての列を含むバッファを返す。公開順を共有している場合は、保存したイベントのバイトを通常の位置に写す。
   *
   * @return blob
   */
  public ByteBuffer blob() {
    materializeEvents();
    return blob;
  }

  /** 通常の提出で読むため、観測席の列を blob 内の標準位置へ写す。 */
  public void materializeEvents() {
    if (eventRows != null) {
      for (int row = 0; row < plan.rows(); row++) {
        if (eventRows[row] == null) continue;
        int first = plan.firstEvent(row);
        copyEvents(
            row,
            0,
            plan.events(row),
            blob,
            plan.offset(Tensor.EVENT_CATEGORY) + first * plan.format().width(Tensor.EVENT_CATEGORY),
            plan.offset(Tensor.EVENT_SCALAR)
                + first * Short.BYTES * plan.format().width(Tensor.EVENT_SCALAR),
            plan.offset(Tensor.RELATION)
                + (plan.tokenStart(Axis.EVENT) + first) * plan.format().width(Tensor.RELATION));
      }
    }
  }

  /**
   * 送るバイト数を返す。
   *
   * @return 計画で使う blob の大きさ
   */
  public int bytes() {
    return plan.bytes();
  }

  /**
   * その手で打牌した後の向聴を、p0 が読む float の値で返す（打牌でない手は 0）。
   *
   * @param action バッチの中の合法手の番号
   * @return 向聴
   */
  public float priorShanten(int action) {
    return priorShanten[Objects.checkIndex(action, plan.count(InputBlob.Axis.ACTION))];
  }

  /**
   * その手で打牌した後の受け入れの残り枚数を、p0 が読む float の値で返す（打牌でない手は 0）。
   *
   * @param action バッチの中の合法手の番号
   * @return 受け入れの枚数
   */
  public float priorUkeire(int action) {
    return priorUkeire[Objects.checkIndex(action, plan.count(InputBlob.Axis.ACTION))];
  }

  /**
   * 行の先頭から数えて、同じ観測席で前の判断までに符号化したイベントの数。{@code [eventPrefix, events)} が、この判断で新しく
   * 符号化したイベント。そのバイトは {@link #copyEvents} で直接読める。{@link #blob} を呼ぶと通常の入力の位置にもそろう。
   * 履歴を使わずに書いた行は 0。
   * 入力の形式が {@link InputFormat#eventsSettleOnPublication} なら、この区間のバイトは前の判断と同じ。
   *
   * @param row バッチの中の行
   */
  public int eventPrefix(int row) {
    return eventPrefix[Objects.checkIndex(row, plan.rows())];
  }

  /** 行の、前の判断から写したイベントの数を書く。呼ぶのは入力のライターだけ。 */
  void eventPrefix(int row, int events) {
    eventPrefix[row] = events;
  }

  /** 合法手ごとの p0 の特徴を書く。呼ぶのは入力のライターだけ。 */
  public void prior(int action, float shanten, float ukeire) {
    priorShanten[action] = shanten;
    priorUkeire[action] = ukeire;
  }
}
