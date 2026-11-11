package com.epsilon.prism.grp;

import com.epsilon.grp.train.GateReport;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.OutcomeGate;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * grp/refits.jsonl の 1 行。学び直しの試行 1 回か、その候補の扱いの変化を表す。扱いの変化には、関門の判定や、学習の JVM が開けなかった
 * 候補の REJECTED がある。同じ id では後の行が前の行に勝つ（{@link #latest}）。
 *
 * <p>候補は、GRP のネットワークと q（局の結果のネットワーク）の組。q を学べなければ、候補は q を持たない。関門を通った候補の q は、
 * q の関門を通った自分の q か、通らなければ今使っている項目から引き継いだ q になる（今使っている項目が q を持たなければ無し）。
 * q のフィールドが無い行は、q を持たない行として読む。候補とその q は run が書いたものなので sha を持たず、sha を持つのは run の外から
 * 引き継いだ q だけ（{@link PhiEntry}）。
 *
 * @param id 試行の id（g&lt;n&gt;-a&lt;k&gt;）。n は再学習の時点、k は試行の番号
 * @param slot 再学習の時点 n。学習の範囲は n まで、関門は n + 1、使い始めは n + 2
 * @param status 状態
 * @param cause 学び直した理由
 * @param window 学習の範囲
 * @param init 学習の始め方
 * @param parent 学習を始めた時点で、n + 1 に使うと決まっていた項目の id（ウォームスタートの元）
 * @param path 候補のディレクトリの絶対パス。学べなかったときは空
 * @param outcomes 候補の q のチェックポイントのディレクトリの絶対パス。持たなければ空
 * @param outcomesSha256 その q の sha。run の外から引き継いだ q のときだけ持ち、ほかは空
 * @param gate 関門の判定。関門の前は null
 * @param outcomeGate q の関門の判定。q を測らなかったときは null
 * @param detail 学習の数や、失敗と拒否の理由
 * @param seconds 学習か関門にかかった秒数
 */
public record RefitRecord(
    String id,
    int slot,
    RefitStatus status,
    RefitCause cause,
    RecencyWindow window,
    GrpInit init,
    String parent,
    String path,
    String outcomes,
    String outcomesSha256,
    GateReport gate,
    OutcomeGate.Report outcomeGate,
    String detail,
    double seconds) {

  /** refits.jsonl のロックの名前。 */
  private static final String LOCK = "refits";

  public RefitRecord {
    outcomes = Objects.requireNonNullElse(outcomes, "");
    outcomesSha256 = Objects.requireNonNullElse(outcomesSha256, "");
    if (outcomes.isEmpty() && !outcomesSha256.isEmpty())
      throw new IllegalArgumentException("An outcome network sha needs the network: " + id);
  }

  /** refits.jsonl の全ての行を、書いた順に返す。ファイルが無ければ空。 */
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

  /** その id の最後の行を返す。無ければ空。 */
  public static Optional<RefitRecord> latest(List<RefitRecord> records, String id) {
    RefitRecord latest = null;
    for (RefitRecord record : records) if (record.id.equals(id)) latest = record;
    return Optional.ofNullable(latest);
  }

  /** 指定した再学習の時点の試行のうち、最後の試行の最後の行を返す。無ければ空。 */
  public static Optional<RefitRecord> latestAt(List<RefitRecord> records, int slot) {
    String id = null;
    for (RefitRecord record : records) if (record.slot == slot) id = record.id;
    return id == null ? Optional.empty() : latest(records, id);
  }

  /** refits.jsonl に 1 行足す。ファイルのロックの中で書き、fsync する。 */
  public void append(GrpFiles files) throws IOException {
    synchronized (PhiHistory.class) {
      try (PhiHistory.Locked locked = PhiHistory.lock(files, LOCK)) {
        PhiHistory.write(files.refits(), PhiHistory.LINES.toJson(this));
      }
    }
  }

  /** 候補が q を持つなら true。 */
  public boolean hasOutcomes() {
    return !outcomes.isEmpty();
  }

  /**
   * 候補から項目を作る。表は重ねず、q を持てば項目に入れる。
   *
   * @param variant 系譜のルール（四人打ちか三人打ちか）
   * @param effectiveFrom 効き始める反復
   * @param reason 理由
   * @return 項目
   */
  public PhiEntry entry(Variant variant, int effectiveFrom, String reason) {
    if (path.isEmpty()) throw new IllegalStateException("Refit attempt " + id + " has no candidate: " + status);
    return new PhiEntry(
        id, PhiEntryKind.NETWORK, variant, path, "", "", "", outcomes, outcomesSha256, effectiveFrom, parent, reason);
  }

  /** 同じ試行の状態を変えた行を返す。q はそのまま引き継ぐ。 */
  public RefitRecord with(RefitStatus next, GateReport report, String why, double took) {
    return new RefitRecord(
        id, slot, next, cause, window, init, parent, path, outcomes, outcomesSha256, report, outcomeGate, why, took);
  }

  /**
   * 同じ試行の q を替えた行を返す。
   *
   * @param directory q のチェックポイントのディレクトリの絶対パス。持たなければ空
   * @param sha その q の sha。run の外から引き継いだ q のときだけ持ち、ほかは空
   * @param report q の関門の判定。測らなかったときは null
   */
  public RefitRecord withOutcomes(String directory, String sha, OutcomeGate.Report report) {
    return new RefitRecord(
        id, slot, status, cause, window, init, parent, path, directory, sha, gate, report, detail, seconds);
  }
}
