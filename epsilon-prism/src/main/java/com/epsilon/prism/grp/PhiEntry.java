package com.epsilon.prism.grp;

import com.epsilon.prism.run.PrismConfig;
import java.util.Objects;

/**
 * grp/phi.jsonl の 1 行（{@link PhiHistory}）。反復 t の Φ は effectiveFrom ≤ t の行のうち effectiveFrom が最も大きい行（同じなら後の行）。
 * 巻き戻しも新しい行を足すだけ。同じ id の行は何度あってもよいが、Φ の中身（{@link #sameValue}）は同じでなければならない。
 *
 * <p>種類ごとの形: 点数だけの Φ は path・sha256・temperatures・calibration が全て空。GRP の項目は path と sha256 が要り、自己対局で学んだ
 * 項目（SELF_CHECKPOINT）は温度も較正の表も重ねない（空）。
 *
 * @param id 項目の id（g000、g&lt;k&gt;-a&lt;n&gt;。固定の出どころは {@value #FIXED_ID}・{@value #SCORE_RANK_ID}・{@value #SCORE_SMOOTH_ID}）
 * @param kind Φ の種類
 * @param path GRP の checkpoint の正確なディレクトリ（点数だけの Φ は空）
 * @param sha256 その checkpoint の sha（{@code EpsilonGrpCheckpointManager.checkpointSha256}。点数だけの Φ は空）
 * @param temperatures 局番号ごとの温度（phiTemperatures の書き方。FIXED_CHECKPOINT だけ、ほかは空）
 * @param calibration 較正の表のパス（FIXED_CHECKPOINT だけ、ほかは空）
 * @param effectiveFrom 効き始める反復（1 以上）
 * @param parent 学習の始めにした項目の id（白紙と g000 は空）
 * @param reason 書いた理由（g000・grp-switch・AUTO・巻き戻し・境目の点検の拒否など）
 */
public record PhiEntry(
    String id,
    PhiEntryKind kind,
    String path,
    String sha256,
    String temperatures,
    String calibration,
    int effectiveFrom,
    String parent,
    String reason) {

  /** phiSource = FIXED_GRP の項目の id。 */
  public static final String FIXED_ID = "fixed";

  /** phiSource = SCORE_RANK の項目の id。 */
  public static final String SCORE_RANK_ID = "score-rank";

  /** phiSource = SCORE_SMOOTH の項目の id。 */
  public static final String SCORE_SMOOTH_ID = "score-smooth";

  /** SELF_GRP の最初の項目の id。 */
  public static final String INITIAL_ID = "g000";

  public PhiEntry {
    Objects.requireNonNull(kind, "kind");
    for (String text : new String[] {id, path, sha256, temperatures, calibration, parent, reason})
      Objects.requireNonNull(text, "Φ entry fields must not be null");
    if (id.isBlank()) throw new IllegalArgumentException("Φ entry without an id");
    if (effectiveFrom < 1)
      throw new IllegalArgumentException("Φ entry " + id + " effectiveFrom must be positive: " + effectiveFrom);
    if (kind.scoreBased()) {
      if (!path.isEmpty() || !sha256.isEmpty() || !temperatures.isEmpty() || !calibration.isEmpty())
        throw new IllegalArgumentException("A score-only Φ entry takes no checkpoint, temperatures or table: " + id);
    } else {
      if (path.isEmpty() || sha256.isEmpty())
        throw new IllegalArgumentException("A GRP Φ entry needs the exact checkpoint and its sha: " + id);
      if (kind == PhiEntryKind.SELF_CHECKPOINT && (!temperatures.isEmpty() || !calibration.isEmpty()))
        throw new IllegalArgumentException("A self-play GRP takes no temperatures or table: " + id);
    }
  }

  /**
   * phiSource = FIXED_GRP の項目（phi.jsonl には書かない。全ての反復で効く）。
   *
   * @param grp 開いた元の GRP の正確なディレクトリ
   * @param sha256 その sha
   */
  public static PhiEntry fixed(PrismConfig config, String grp, String sha256) {
    return new PhiEntry(
        FIXED_ID,
        PhiEntryKind.FIXED_CHECKPOINT,
        grp,
        sha256,
        config.phiTemperatures(),
        config.phiCalibration(),
        1,
        "",
        "phiSource = FIXED_GRP");
  }

  /** phiSource = SCORE_RANK の項目（phi.jsonl には書かない。全ての反復で効く）。 */
  public static PhiEntry scoreRank() {
    return new PhiEntry(
        SCORE_RANK_ID, PhiEntryKind.SCORE_RANK, "", "", "", "", 1, "", "phiSource = SCORE_RANK");
  }

  /** phiSource = SCORE_SMOOTH の項目（phi.jsonl には書かない。全ての反復で効く）。 */
  public static PhiEntry scoreSmooth() {
    return new PhiEntry(
        SCORE_SMOOTH_ID, PhiEntryKind.SCORE_SMOOTH, "", "", "", "", 1, "", "phiSource = SCORE_SMOOTH");
  }

  /** 同じ Φ（種類・checkpoint・sha・温度・較正の表が同じ。id・効き始め・親・理由は見ない）なら true。 */
  public boolean sameValue(PhiEntry other) {
    return kind == other.kind
        && path.equals(other.path)
        && sha256.equals(other.sha256)
        && temperatures.equals(other.temperatures)
        && calibration.equals(other.calibration);
  }

  /** 同じ項目を反復 from から効かせる行（巻き戻しと、境目の点検で拒んだ候補の代わり）。 */
  public PhiEntry from(int from, String why) {
    return new PhiEntry(id, kind, path, sha256, temperatures, calibration, from, parent, why);
  }
}
