package com.epsilon.agent.observe;

import com.epsilon.agent.observe.ObservationSchema.DiscardBits;
import com.epsilon.agent.observe.ObservationSchema.EventBits;
import com.epsilon.agent.observe.ObservationSchema.MeldBits;
import com.epsilon.mahjong.engine.EventKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;

/**
 * エンジンが記録した公開のイベント（EventLog）を、番号の順に、記録 O のイベントのセクションへ書く。これまでの経過を河や手から
 * 組み立て直さず、エンジンの記録を読む。
 *
 * <p>書き方は 2 つ。{@link #write} は全てのイベントを書く。{@link #append} は 1 本の経路（同じエンジンを進める局面の列）の続きとして、
 * 観測席ごとに前に書いたセクションを持ち、増えたイベントだけを書き足してセクションを丸ごと写す。局の中でエンジンの記録は追記するだけで、
 * 書き換わるのは最後の打牌の鳴いた席だけなので、前に書いた最後のイベントも書き直す。局が変わったら、経路が {@link #reset} を呼ぶ。
 * 局面を写して分けた経路は、親のセクションの写しから始める（{@link #EventRecords(EventRecords)}）。
 * スレッドセーフではない。
 *
 * <p>席は観測席から見た相対席で書く。鳴いた席も相対席で書き、鳴かれていなければ −1 にする。イベントの番号は記録の中の位置
 * そのものなので、フィールドには書かない。ポンのイベントは、後で加槓してもポンの面子のまま書く。加槓は、O の副露のイベントとして
 * 加槓の面子を書く。鳴かれた打牌と副露が合っているか（鳴きは元の打牌の直後か、加槓は自分のポンからか）は、エンジンが記録を書く
 * ときに確かめる。北抜きは種類と席だけを書く。各イベントは {@code long} に詰める。公開順は観測席の末尾に追記し、共有中の末尾に続けるときだけ区間を分ける。
 */
public final class EventRecords {

  private final ObservationSchema schema;
  private final int seats;
  private final int discard;
  private final int meld;
  private final int nuki;

  /** セクションを書いた時点の公開順。鳴いた席は後で変わるので、その値を含む計算には使わない。 */
  public static final class Prefix {
    /** 分岐や判断と共有して固定した直前の区間。局の最初は null。 */
    public final Prefix parent;

    /** エンジンのイベントの個数。追加の槓ドラ表示を含まない。 */
    public int records;

    /** 公開順の終端。追加の槓ドラ表示を含む。 */
    public int end;

    /** 接頭辞を使う最後の経路・判断が終わったときに、計算結果の保存先へ知らせる。 */
    public interface Placement {
      void unused();
    }

    private static final AtomicIntegerFieldUpdater<Prefix> USERS =
        AtomicIntegerFieldUpdater.newUpdater(Prefix.class, "users");
    private volatile int users = 1;
    private volatile Placement placement;

    /** 入力形式がこの区間に追記した符号。GPU の保存先とは独立に、同じ公開順を使う経路が共有する。 */
    public interface Encoding {
      void unused();
    }

    private volatile Encoding encoding;

    public Encoding encoding() {
      return encoding;
    }

    public void encoding(Encoding value) {
      encoding = value;
    }

    /** 末尾を持つ経路・判断と、生きている直後の区間からの参照数。 */
    public int references() {
      return users;
    }

    /** 生きている区間をもう 1 つの経路または判断に渡す。 */
    public void retain() {
      int count;
      do {
        count = users;
        if (count <= 0) throw new IllegalStateException("Cannot retain a released event prefix");
      } while (!USERS.compareAndSet(this, count, Math.incrementExact(count)));
    }

    /** 所有を返す。子がなくなれば親に対する参照も返す。 */
    public void release() {
      for (Prefix p = this; p != null; p = p.parent) {
        int remaining = USERS.decrementAndGet(p);
        if (remaining > 0) return;
        if (remaining < 0) throw new IllegalStateException("Event prefix released twice");
        synchronized (p) {
          if (p.placement != null) p.placement.unused();
          if (p.encoding != null) p.encoding.unused();
          p.placement = null;
          p.encoding = null;
        }
      }
    }

    /** 推論の保存先。入れ替えはこの区間をロックして行う。 */
    public Placement placement() {
      return placement;
    }

    public void place(Placement next) {
      placement = next;
    }

    private Prefix(Prefix parent, int records, int end) {
      if (parent != null) parent.retain();
      this.parent = parent;
      this.records = records;
      this.end = end;
    }

    /**
     * 公開イベントが増えたときは、新しい末尾の参照を 1 つ渡す。増えていなければ元の末尾をそのまま返す。
     * 局の最初と単独の O の再生では親を渡さない。
     */
    public static Prefix append(Prefix parent, int records, int kanIndicators) {
      int end = Math.addExact(records, kanIndicators);
      if (records < 0
          || kanIndicators < 0
          || parent != null
              && (records < parent.records || kanIndicators < parent.end - parent.records)) {
        throw new IllegalArgumentException("Public event prefix shrank");
      }
      return end == (parent == null ? 0 : parent.end) ? parent : new Prefix(parent, records, end);
    }

    /** 経路だけが所有する末尾にはそのまま追記する。分岐・未完の判断から見える末尾は固定する。 */
    private Prefix extend(int nextRecords, int indicators) {
      if (nextRecords < records || indicators < end - records) {
        throw new IllegalArgumentException("Public event prefix shrank");
      }
      if (users != 1) return append(this, nextRecords, indicators);
      end = Math.addExact(nextRecords, indicators);
      records = nextRecords;
      return this;
    }

    /** この区間の先頭。槓ドラを含む公開イベントの番号で返す。 */
    public int first() {
      return parent == null ? 0 : parent.end;
    }
  }

  private static final class Section {
    final long[] words;
    int written;
    Prefix tip;

    Section(int longs) {
      words = new long[longs];
    }

    Section(Section parent) {
      words = parent.words.clone();
      written = parent.written;
      tip = parent.tip;
      if (tip != null) tip.retain();
    }
  }

  private final Section[] sections;

  public EventRecords(Variant variant) {
    schema = ObservationSchema.of(variant);
    seats = variant.seats();
    discard = schema.eventKinds.index(ObservationSchema.EventKind.DISCARD);
    meld = schema.eventKinds.index(ObservationSchema.EventKind.MELD);
    nuki =
        schema.eventKinds.contains(ObservationSchema.EventKind.NUKI)
            ? schema.eventKinds.index(ObservationSchema.EventKind.NUKI)
            : -1;
    sections = new Section[seats];
    int longs = schema.event.longs(schema.header.max(ObservationSchema.HeaderBits.EVENTS));
    for (int seat = 0; seat < seats; seat++) sections[seat] = new Section(longs);
  }

  /**
   * 写した局面から始める経路のセクションを、親の経路のセクションの写しで作る。写した局面のエンジンの記録は親の記録と同じなので、続きを
   * 書き足せる。
   *
   * @param parent 写した元の経路のセクション
   */
  public EventRecords(EventRecords parent) {
    schema = parent.schema;
    seats = parent.seats;
    discard = parent.discard;
    meld = parent.meld;
    nuki = parent.nuki;
    sections = new Section[seats];
    for (int seat = 0; seat < seats; seat++) sections[seat] = new Section(parent.sections[seat]);
  }

  /** 経路が次の局に入った。全ての観測席のセクションを捨てる。 */
  public void reset() {
    for (Section section : sections) {
      Arrays.fill(section.words, 0, schema.event.longs(section.written), 0L);
      section.written = 0;
      if (section.tip != null) section.tip.release();
      section.tip = null;
    }
  }

  /**
   * エンジンの記録にある全てのイベントを、記録 O のイベントのセクションに書く。経路のセクションは使わない。
   *
   * @param log エンジンが記録した公開のイベント
   * @param observer 観測席（絶対席）
   * @param words 書く先の配列（イベントのセクションは 0 で埋めてあること）
   * @param at 書く先の配列で、イベントのセクションが始まる添字
   */
  void write(EventLog log, int observer, long[] words, int at) {
    for (int event = 0, events = log.size(); event < events; event++) {
      schema.event.putRecord(words, at, event, record(log, event, observer));
    }
  }

  /**
   * {@link #write} と同じセクションを、経路の続きとして作る。観測席のセクションに、前に書いた最後のイベントと増えたイベントを書いてから
   * 写す。
   *
   * @throws IllegalStateException 前に書いたときよりイベントが少ない（局が変わったのに {@link #reset} が呼ばれていない）とき
   */
  void append(EventLog log, int observer, long[] words, int at) {
    Section section = sections[observer];
    int events = log.size(), from = section.written;
    if (events < from) {
      throw new IllegalStateException(
          "Event log shrank from " + from + " to " + events + " without a reset");
    }
    for (int event = Math.max(0, from - 1); event < events; event++) {
      schema.event.putRecord(section.words, 0, event, record(log, event, observer));
    }
    section.written = events;
    System.arraycopy(section.words, 0, words, at, schema.event.longs(events));
  }

  /** 終わった経路の各観測席が所有する公開区間を返す。 */
  public void releasePrefixes() {
    for (Section section : sections) {
      if (section.tip != null) section.tip.release();
      section.tip = null;
    }
  }

  /** 1 つのイベントの記録を {@code long} に詰める。 */
  private long record(EventLog log, int event, int observer) {
    int author = Math.floorMod(log.seat(event) - observer, seats);
    EventKind kind = log.kind(event);
    return switch (kind) {
      case DISCARD -> {
        int calledBy = log.calledBy(event);
        long word = head(discard, author);
        word = schema.discard.with(word, DiscardBits.TILE, log.tile(event));
        word = schema.discard.with(word, DiscardBits.AKA, log.red(event) ? 1 : 0);
        word = schema.discard.with(word, DiscardBits.TSUMOGIRI, log.tsumogiri(event) ? 1 : 0);
        word = schema.discard.with(word, DiscardBits.RIICHI, log.riichi(event) ? 1 : 0);
        yield schema.discard.with(
            word,
            DiscardBits.CALLED_BY,
            calledBy == EventLog.NONE ? EventLog.NONE : Math.floorMod(calledBy - observer, seats));
      }
      case MELD, KAKAN -> schema.meld.with(head(meld, author), MeldBits.MELD, log.meld(event));
      case NUKI -> head(nuki, author);
    };
  }

  /** 書き終えた観測席のセクションに、槓ドラの表示を含む公開順を結びつける。 */
  public Prefix prefix(int observer, int kanIndicators) {
    Section section = sections[observer];
    Prefix previous = section.tip;
    section.tip =
        previous == null
            ? Prefix.append(null, section.written, kanIndicators)
            : previous.extend(section.written, kanIndicators);
    if (previous != null && previous != section.tip) previous.release();
    return section.tip;
  }

  /** イベントの記録のうち、どの種類にも共通する先頭部分（種類と、イベントを起こした席）を詰めた {@code long}。 */
  private long head(int kind, int author) {
    return schema.event.with(schema.event.with(0L, EventBits.KIND, kind), EventBits.AUTHOR, author);
  }
}
