package com.epsilon.prism.grp;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * grp/refits.jsonl の 1 行: GRP の学習の試行 1 回か、その候補の扱いの変化（境目の点検の REJECTED、AUTO の STALE）。同じ id の後の
 * 行が前の行に勝つ（{@link #latest}）。
 *
 * @param id 試行の id（g&lt;k&gt;-a&lt;n&gt;。k は窓の取り置きのコホート、n は試行の番号）
 * @param status 状態
 * @param cause 学習した理由
 * @param window 学習の窓と取り置き（窓が取れなければ null）
 * @param init 学習の始め（使用中が点数だけの Φ なら SCRATCH）
 * @param parent 使用中の項目の id（関門の相手と温めの始め）
 * @param path 公開した候補の正確なディレクトリ（READY のときだけ、ほかは空）
 * @param sha256 候補の sha（READY のときだけ、ほかは空）
 * @param gate 関門の報告（学習まで届かなければ null）
 * @param detail 失敗や拒否の理由（なければ空）
 * @param seconds 学習と関門の秒数
 */
public record RefitRecord(
    String id,
    RefitStatus status,
    RefitCause cause,
    RefitWindow window,
    GrpInit init,
    String parent,
    String path,
    String sha256,
    GateReport gate,
    String detail,
    double seconds) {

  /** refits.jsonl の錠の名前。 */
  private static final String LOCK = "refits";

  /** refits.jsonl の全ての行（書いた順。ファイルがなければ空）。 */
  public static List<RefitRecord> readAll(GrpFiles files) throws IOException {
    List<RefitRecord> records = new ArrayList<>();
    List<String> lines = PhiHistory.lines(files.refits());
    for (int i = 0; i < lines.size(); i++) {
      RefitRecord record;
      try {
        record = PhiHistory.LINES.fromJson(lines.get(i), RefitRecord.class);
      } catch (RuntimeException broken) {
        throw new IOException("Broken line " + (i + 1) + " of " + files.refits(), broken);
      }
      if (record.id == null || record.status == null)
        throw new IOException("Line " + (i + 1) + " of " + files.refits() + " has no id or status");
      records.add(record);
    }
    return records;
  }

  /** id の最後の行（なければ空）。 */
  public static Optional<RefitRecord> latest(List<RefitRecord> records, String id) {
    RefitRecord latest = null;
    for (RefitRecord record : records) if (record.id.equals(id)) latest = record;
    return Optional.ofNullable(latest);
  }

  /** refits.jsonl に 1 行足す（ファイルの錠の中で書いて fsync する）。 */
  public void append(GrpFiles files) throws IOException {
    synchronized (PhiHistory.class) {
      try (PhiHistory.Locked locked = PhiHistory.lock(files, LOCK)) {
        PhiHistory.write(files.refits(), PhiHistory.LINES.toJson(this));
      }
    }
  }

  /** READY の候補の項目（kind は SELF_CHECKPOINT、温度と較正は空）。 */
  public PhiEntry entry(int effectiveFrom, String reason) {
    if (status != RefitStatus.READY)
      throw new IllegalStateException("Refit attempt " + id + " is " + status + ", not READY");
    return new PhiEntry(id, PhiEntryKind.SELF_CHECKPOINT, path, sha256, "", "", effectiveFrom, parent, reason);
  }

  /** 同じ試行の状態を変えた行（境目の点検の REJECTED など）。 */
  public RefitRecord withStatus(RefitStatus next, String why) {
    return new RefitRecord(id, next, cause, window, init, parent, path, sha256, gate, why, seconds);
  }
}
