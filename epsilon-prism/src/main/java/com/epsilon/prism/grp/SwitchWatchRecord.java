package com.epsilon.prism.grp;

import com.epsilon.grp.train.SwitchWatchLine;
import com.epsilon.grp.train.SwitchWatchVerdict;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * grp/switch-watch.jsonl の 1 行。GRP を切り替えた後の監視で、確定したコホートの区切りごとに 1 行書く。この行は、その区切りの
 * 監視が済んだことを示す完了マーカーも兼ねる（行の後に行う候補の関門が済んだかは、refits.jsonl の候補の状態で見分ける）。監視の行と判定、
 * 判定の後の状態を持つ。書き込むのは学習の JVM の {@link GrpBackground} だけ。
 *
 * @param cohort 確定したコホート h
 * @param usedId h を集めたときに使った項目の id
 * @param previousId その前に使っていた別の項目の id。無ければ空
 * @param line 監視の行。1 つ前の項目か h の記録が無ければ null
 * @param verdict 判定。重ならない 8 行がそろってこの行で判定したときだけ持ち、ほかは null
 * @param widened この行の後で τ を 2 倍にしているか
 * @param halted この行の後で学び直しを止めているか
 * @param rollbackTo ROLLBACK で戻した先の項目の id。戻さなければ空
 */
public record SwitchWatchRecord(
    int cohort,
    String usedId,
    String previousId,
    SwitchWatchLine line,
    SwitchWatchVerdict verdict,
    boolean widened,
    boolean halted,
    String rollbackTo) {

  /**
   * 監視の今の状態。
   *
   * @param lastCohort 処理を済ませた最後の区切りのコホート。無ければ 0
   * @param pending 最後の判定より後の監視の行。コホートの昇順に並ぶ
   * @param widened τ を 2 倍にしているか
   * @param halted 学び直しを止めているか
   */
  record State(int lastCohort, List<SwitchWatchLine> pending, boolean widened, boolean halted) {}

  /** switch-watch.jsonl の全ての行を書いた順に返す。ファイルが無ければ空。 */
  static List<SwitchWatchRecord> readAll(GrpFiles files) throws IOException {
    List<SwitchWatchRecord> records = new ArrayList<>();
    List<String> lines = PhiHistory.lines(files.switchWatch());
    for (int i = 0; i < lines.size(); i++) {
      try {
        records.add(PhiHistory.LINES.fromJson(lines.get(i), SwitchWatchRecord.class));
      } catch (RuntimeException broken) {
        throw new IOException("Broken line " + (i + 1) + " of " + files.switchWatch(), broken);
      }
    }
    return records;
  }

  /** 全ての行から今の状態を求める。 */
  static State state(List<SwitchWatchRecord> records) {
    int last = 0;
    boolean widened = false;
    boolean halted = false;
    List<SwitchWatchLine> pending = new ArrayList<>();
    for (SwitchWatchRecord record : records) {
      last = Math.max(last, record.cohort);
      if (record.line != null) pending.add(record.line);
      if (record.verdict != null) pending.clear();
      widened = record.widened;
      halted = record.halted;
    }
    return new State(last, pending, widened, halted);
  }

  /** switch-watch.jsonl に 1 行足して fsync する。 */
  void append(GrpFiles files) throws IOException {
    PhiHistory.write(files.switchWatch(), PhiHistory.LINES.toJson(this));
  }
}
