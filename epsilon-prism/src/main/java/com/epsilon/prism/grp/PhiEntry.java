package com.epsilon.prism.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * grp/phi.jsonl の 1 行（{@link PhiHistory}）。反復 t で効くのは、効き始めが t 以下の行のうち効き始めが最も遅い行（同じなら後に書いた行）。
 * 前の項目に戻すときも新しい行を足すだけにする。同じ id の行が何度出てもよいが、Φ の中身（{@link #sameValue}）は同じでなければならない。
 *
 * <p>sha を持つのは run の外から来たファイル（g000 と固定の GRP のチェックポイント・表・q）で、開くたびに照合する
 * （{@link HandEndPhi}）。run が書いた候補とその q は書き直さないので、sha を持たず、id で見分ける。
 *
 * <p>種類ごとの決まり: 点数だけの Φ では、チェックポイント・表・q のパスと sha が全て空。NETWORK ではチェックポイントのパスが要り、
 * 候補でない項目はその sha も要る。表は中身が GruSinkhorn のときだけ重ね、表のパスと sha は両方あるか両方空かのどちらか（空なら重ねない）。
 * 自己対局で学んだ候補（id が g&lt;k&gt;-a&lt;n&gt;）は表を重ねない。q（局の結果のネットワーク）は LATE のモードで最後の局を数え上げる
 * ときだけ読む。候補でない項目の q は sha を持ち、候補の q は run の外の q を引き継いだときだけ sha を持つ。数え上げの葉は GrpNetwork
 * なので、q を持つ項目は表を重ねない。q のパスが無い行は、q を持たない行として読む。id にはモードの別名（{@link PhiMode#alias}）を
 * 使えない。
 *
 * @param id 項目の id（g000 か g&lt;k&gt;-a&lt;n&gt;。固定の出どころなら {@value #FIXED_ID}・{@value #SCORE_RANK_ID}・{@value #SCORE_SMOOTH_ID}）
 * @param kind Φ の種類
 * @param variant ルール（収集のルールと違えば開けない）
 * @param path GRP のチェックポイントそのもののディレクトリ（点数だけの Φ では空）
 * @param sha256 そのチェックポイントの sha（点数だけの Φ と、run が書いた候補では空）
 * @param table 重ねる PhiTable のファイル（重ねなければ空）
 * @param tableSha256 そのファイルの sha（重ねなければ空）
 * @param outcomes q のチェックポイントそのもののディレクトリ（持たなければ空）
 * @param outcomesSha256 その q の sha（q を持たないときと、run が書いた q では空）
 * @param effectiveFrom 効き始める反復（1 以上）
 * @param parent 関門で比べた相手の項目の id（自己対局の候補だけ。ほかは空）
 * @param reason 書いた理由（g000・grp-switch・AUTO・戻し・開けなかった候補の代わりなど）
 */
public record PhiEntry(
    String id,
    PhiEntryKind kind,
    Variant variant,
    String path,
    String sha256,
    String table,
    String tableSha256,
    String outcomes,
    String outcomesSha256,
    int effectiveFrom,
    String parent,
    String reason) {

  /** Φ の出どころが固定の GRP のときの項目の id。 */
  public static final String FIXED_ID = "fixed";

  /** Φ の出どころが点数の順位のときの項目の id。 */
  public static final String SCORE_RANK_ID = "score-rank";

  /** Φ の出どころが点数のなめらかな関数のときの項目の id。 */
  public static final String SCORE_SMOOTH_ID = "score-smooth";

  /** 自分の GRP の最初の項目の id。 */
  public static final String INITIAL_ID = "g000";

  /** 戻しの行で、理由の先頭に付ける文字列。 */
  static final String ROLLBACK = "rollback:";

  /** 自己対局で学んだ候補の id の形（g&lt;学び直した時点&gt;-a&lt;試行の番号&gt;）。 */
  static final Pattern CANDIDATE = Pattern.compile("g(\\d+)-a(\\d+)");

  public PhiEntry {
    Objects.requireNonNull(kind, "kind");
    Objects.requireNonNull(variant, "variant");
    outcomes = Objects.requireNonNullElse(outcomes, "");
    outcomesSha256 = Objects.requireNonNullElse(outcomesSha256, "");
    for (String text : new String[] {id, path, sha256, table, tableSha256, parent, reason})
      Objects.requireNonNull(text, "Φ entry fields must not be null");
    if (id.isBlank()) throw new IllegalArgumentException("Φ entry without an id");
    if (PhiMode.byAlias(id).isPresent()) throw new IllegalArgumentException("Φ entry id " + id + " names a mode");
    if (effectiveFrom < 1)
      throw new IllegalArgumentException("Φ entry " + id + " effectiveFrom must be positive: " + effectiveFrom);
    if (table.isEmpty() != tableSha256.isEmpty())
      throw new IllegalArgumentException("A Φ table needs its sha and the sha needs a table: " + id);
    if (outcomes.isEmpty() && !outcomesSha256.isEmpty())
      throw new IllegalArgumentException("An outcome network sha needs the network: " + id);
    if (kind.scoreBased()) {
      if (!path.isEmpty() || !sha256.isEmpty() || !table.isEmpty() || !outcomes.isEmpty())
        throw new IllegalArgumentException("A score-only Φ entry takes no checkpoint, table or outcome network: " + id);
    } else if (path.isEmpty()) {
      throw new IllegalArgumentException("A GRP Φ entry needs the exact checkpoint: " + id);
    } else if (!CANDIDATE.matcher(id).matches()
        && (sha256.isEmpty() || outcomes.isEmpty() != outcomesSha256.isEmpty())) {
      throw new IllegalArgumentException("An outside GRP Φ entry needs the sha of its checkpoint and q: " + id);
    }
    if (!outcomes.isEmpty() && !table.isEmpty())
      throw new IllegalArgumentException("An entry with an outcome network takes no table: " + id);
  }

  /** Φ の出どころが点数だけで決まる固定の Φ のときの項目。phi.jsonl には書かず、全ての反復で効く。 */
  public static PhiEntry scoreOnly(PhiEntryKind kind, Variant variant) {
    String id = switch (kind) {
      case SCORE_RANK -> SCORE_RANK_ID;
      case SCORE_SMOOTH -> SCORE_SMOOTH_ID;
      case NETWORK -> throw new IllegalArgumentException("Not a score-only kind: " + kind);
    };
    return new PhiEntry(id, kind, variant, "", "", "", "", "", "", 1, "", "phi.source = " + kind);
  }

  /** 自己対局で学んだ候補なら true。 */
  public boolean candidate() {
    return CANDIDATE.matcher(id).matches();
  }

  /** 戻しの行なら true。 */
  public boolean rollback() {
    return reason.startsWith(ROLLBACK);
  }

  /** q（局の結果のネットワーク）を持つなら true。 */
  public boolean hasOutcomes() {
    return !outcomes.isEmpty();
  }

  /**
   * 同じ Φ を表すなら true。種類・ルール・チェックポイントのパスと sha・表の sha・q の sha を比べ、sha を持たない q（run が書いた q）は
   * パスで比べる。id・効き始め・親・理由・表のパス・sha を持つ q のパスは見ない。
   */
  public boolean sameValue(PhiEntry other) {
    return kind == other.kind
        && variant == other.variant
        && path.equals(other.path)
        && sha256.equals(other.sha256)
        && tableSha256.equals(other.tableSha256)
        && outcomesSha256.equals(other.outcomesSha256)
        && (!outcomesSha256.isEmpty() || outcomes.equals(other.outcomes));
  }

  /** 同じ項目（q も同じ）を反復 {@code from} から効かせる行を作る。前の項目に戻すときと、開けなかった候補の代わりに使う。 */
  public PhiEntry from(int from, String why) {
    return new PhiEntry(id, kind, variant, path, sha256, table, tableSha256, outcomes, outcomesSha256, from, parent, why);
  }
}
