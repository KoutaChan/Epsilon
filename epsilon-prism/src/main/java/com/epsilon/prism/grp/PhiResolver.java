package com.epsilon.prism.grp;

import com.epsilon.grp.Phi;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習の JVM の中で、反復ごとの Φ を決める。{@code TrainingResources} が持つ。反復 t の Φ は、t に効く GRP の項目に t に効くモードを
 * 当てた値（{@link PhiValue}）になる。
 *
 * <p>GRP の項目は、自分の GRP なら phi.jsonl から、固定の GRP なら反復に固定した設定の行のチェックポイントと q から、点数だけの Φ なら
 * 出どころから決まる。モードは grp/mode.jsonl から決まり、点数だけの Φ ではいつも GRP のモードになる。
 *
 * <p>反復 t の始めに、phi.jsonl と mode.jsonl を同じロックの中で 1 回ずつ読む。値が今開いている値と違えば（{@link PhiValue#sameValue}）、
 * {@link HandEndPhi#open} で新しい Phi を開く。新しい Phi は終盤の共有メモも新しく持つので、メモが別の値の Φ を返すことはない。
 * チェックポイント・表・q は項目に書いたパスで読み、run の外から来たファイルは sha を照合する。反復 t の項目は反復 t − 1 の後処理が
 * 済んでから決まるので、呼び出し元は先に {@link GrpBackground#awaitBoundary} で待つこと。
 *
 * <p>反復 t で初めて効く自己対局の候補が開けないときは、拒んで前の項目で続ける。開けないのは、読めないとき（LATE のモードでは q も
 * 読む）、run の外から引き継いだ q の sha が違うとき、READY でないとき。このとき refits.jsonl に REJECTED を書き、前の項目を反復 t
 * から効かせる行を phi.jsonl に足す。行を足すので、再開しても同じ履歴になる。既に効いたことのある項目や固定の GRP が開けなければ
 * 止める。黙って別の Φ やモードに切り替えることはしない。
 *
 * <p>学習の JVM が動いていることは、呼び出し元が持つ run の学習のロックで示す（{@link PhiHistory#firstOpenIteration}）。
 */
public final class PhiResolver implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(PhiResolver.class);

  private final PhiSetup setup;
  private final GrpFiles files;

  /** LATE のモードで、数え上げた卓の Φ を覚えておく共有メモのスロット数。LATE のモードを持たない点数だけの Φ では 0。 */
  private final int lateSlots;

  /** 今開いている Φ。まだ開いていなければ null。 */
  private ActivePhi opened;

  private PhiResolver(PhiSetup setup, GrpFiles files) {
    this.setup = setup;
    this.files = files;
    lateSlots = setup.phi().lateSettings().map(LateSettings::memo).orElse(0);
  }

  /**
   * run の Φ の出どころに合わせて作る。自分の GRP なら、phi.jsonl が無ければ g000 を書く（{@link PhiHistory#initialize}）。自分の GRP と
   * 固定の GRP では、LATE のモードを使い始める反復が正で mode.jsonl に行が無ければ、初めの LATE の行を書く。Φ はまだ開かない。
   *
   * @param setup JVM を起動したときの設定から取った、Φ の出どころ・ルール・終盤のメモの大きさ
   * @throws IOException phi.jsonl か mode.jsonl が読めないか書けないとき
   */
  public static PhiResolver open(PhiSetup setup, GrpFiles files) throws IOException {
    switch (setup.phi()) {
      case PhiSettings.SelfGrp self -> {
        PhiEntry initial = PhiHistory.initialize(files, setup);
        LOG.info("Φ history {} starts with {}", files.phiHistory(), initial);
      }
      case PhiSettings.FixedGrp fixed -> PhiHistory.initializeModes(files, fixed.late());
      case PhiSettings.ScoreRank rank ->
          LOG.warn("The Φ source is {}: hand ends are valued by the scores, not a GRP", PhiSource.SCORE_RANK);
      case PhiSettings.ScoreSmooth smooth ->
          LOG.warn("The Φ source is {}: hand ends are valued by the scores, not a GRP", PhiSource.SCORE_SMOOTH);
    }
    return new PhiResolver(setup, files);
  }

  /**
   * 反復 t の Φ を返す。値が変わっていれば開き直す。開けなかった候補は拒み、前の項目を効かせ直してから決め直す。
   *
   * @param iteration 反復に固定した設定の行の、Φ のセクション。固定の GRP は、この行のチェックポイントと q で開く
   * @throws IOException 既に効いたことのある項目か固定の GRP が開けないとき。系譜はここで止まる
   */
  public ActivePhi resolve(int t, PhiSettings iteration) throws IOException {
    boolean scoreOnly = iteration instanceof PhiSettings.ScoreRank || iteration instanceof PhiSettings.ScoreSmooth;
    PhiHistory.Lines lines = scoreOnly ? new PhiHistory.Lines(List.of(), List.of()) : PhiHistory.readLocked(files);
    PhiEntry current = entry(t, iteration, lines.entries());
    PhiValue value =
        ModeLine.effectiveAt(lines.modes(), t)
            .map(line -> PhiValue.of(current, line))
            .orElseGet(() -> PhiValue.grp(current));
    if (opened == null || !opened.value().sameValue(value)) {
      boolean fresh = fresh(lines.entries(), current, t);
      Phi next;
      try {
        next = open(value, fresh);
      } catch (IOException | RuntimeException failure) {
        if (!fresh)
          throw new IOException(
              "Φ entry " + current.id() + " in the " + value.mode() + " mode in effect at iteration " + t
                  + " cannot be opened",
              failure);
        reject(lines.entries(), current, t, failure);
        return resolve(t, iteration);
      }
      opened = new ActivePhi(value, next);
      LOG.info("t={} Φ is now {}", t, value);
    }
    if (value.requested() != value.mode())
      LOG.warn(
          "t={} the {} mode needs an outcome network, which Φ entry {} lacks: the hand ends take the {} mode",
          t, value.requested(), current.id(), value.mode());
    return new ActivePhi(value, opened.phi());
  }

  /**
   * 記録した値の Φ を開く。Φ の段の完了マーカーに書いた値で集め直すときに使う。今開いている値と同じなら、開き直さずにそれを使う。
   *
   * @throws IOException 値のチェックポイントか表か q が読めないか、run の外から来たファイルの sha が合わないとき
   */
  public ActivePhi reopen(PhiValue value) throws IOException {
    if (opened == null || !opened.value().sameValue(value)) {
      opened = new ActivePhi(value, HandEndPhi.open(value, setup.utility(), lateSlots));
    }
    return new ActivePhi(value, opened.phi());
  }

  /** 開いた Φ を解放する。Φ は純 Java で、閉じるリソースは持たない。 */
  @Override
  public void close() {
    opened = null;
  }

  /** 反復 t に使う GRP の項目を返す。 */
  private PhiEntry entry(int t, PhiSettings iteration, List<PhiEntry> history) throws IOException {
    return switch (iteration) {
      case PhiSettings.SelfGrp self ->
          PhiHistory.effectiveAt(history, t)
              .orElseThrow(() -> new IOException("No Φ entry is in effect at iteration " + t + " in " + files.phiHistory()));
      case PhiSettings.FixedGrp fixed -> PhiScorer.original(new PhiSetup(iteration, setup.variant(), setup.seed()));
      case PhiSettings.ScoreRank rank -> PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, setup.variant());
      case PhiSettings.ScoreSmooth smooth -> PhiEntry.scoreOnly(PhiEntryKind.SCORE_SMOOTH, setup.variant());
    };
  }

  /** 項目が、反復 t で初めて効く自己対局の候補なら true。同じ id の行が t より前から効いていれば false。 */
  private static boolean fresh(List<PhiEntry> entries, PhiEntry current, int t) {
    if (!current.candidate()) return false;
    for (PhiEntry entry : entries)
      if (entry.id().equals(current.id()) && entry.effectiveFrom() < t) return false;
    return true;
  }

  /** 値を開く。初めて効く候補なら、READY であることも確かめる。 */
  private Phi open(PhiValue value, boolean fresh) throws IOException {
    if (fresh) {
      String id = value.entry().id();
      Optional<RefitRecord> record = RefitRecord.latest(RefitRecord.readAll(files), id);
      if (record.isEmpty() || record.get().status() != RefitStatus.READY)
        throw new IOException("Candidate " + id + " is not READY: " + record.map(RefitRecord::status).orElse(null));
    }
    return HandEndPhi.open(value, setup.utility(), lateSlots);
  }

  /**
   * 反復 t で初めて効く候補を拒む。refits.jsonl に REJECTED を書き、前の項目を反復 t から効かせる行を phi.jsonl に足す。refits.jsonl が
   * 読めないか書けなくても、phi.jsonl の行は必ず書く。履歴が実際に使った Φ と合っていれば、系譜は続けられる。
   */
  private void reject(List<PhiEntry> entries, PhiEntry candidate, int t, Throwable failure) throws IOException {
    PhiEntry previous =
        PhiHistory.effectiveAt(entries, t - 1)
            .orElseThrow(() -> new IOException("Φ entry " + candidate.id() + " cannot be opened and nothing precedes it", failure));
    String detail = "could not open at iteration " + t + ": " + failure;
    try {
      Optional<RefitRecord> ready = RefitRecord.latest(RefitRecord.readAll(files), candidate.id());
      if (ready.isPresent()) ready.get().with(RefitStatus.REJECTED, ready.get().gate(), detail, 0).append(files);
    } catch (IOException | RuntimeException unreadable) {
      LOG.warn("t={} could not extend {}: {}", t, files.refits(), unreadable.toString());
    }
    PhiHistory.appendUnstarted(files, previous.from(t, "rejected " + candidate.id() + " at iteration " + t));
    LOG.warn("t={} rejected Φ candidate {} and kept {}: {}", t, candidate.id(), previous.id(), failure.toString());
  }
}
