package com.epsilon.prism.grp;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpModel;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GruSinkhorn;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.Phi;
import com.epsilon.grp.PhiTable;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Φ の項目から Phi を開く唯一の入口。系譜の収集・裏の採点・関門・切り替えの後の監視・check-config はどれもここを通るので、裏で採点した値は
 * ラベルに入った値と同じになる。項目が sha を持つファイル（run の外から来たチェックポイント・表・q）は開くたびに sha を照合し、ルールが
 * 項目と違えば開かない。run が書いた候補とその q は書き直さないので、照合しない。LATE のモードでは、項目のネットワークを数え上げの葉に
 * し、q で最後の局を数え上げる（{@link Phi#late}）。葉は GrpNetwork でなければならない。
 */
public final class HandEndPhi {

  private HandEndPhi() {}

  /**
   * 項目の GRP のモードの Phi を開く。
   *
   * @throws IOException チェックポイントか表が読めないか、sha が項目と違うとき
   * @throws IllegalStateException チェックポイントか表のルールが項目と違うとき、または GruSinkhorn 以外のモデルに表を重ねようとしたとき
   */
  public static Phi open(PhiEntry entry, RankUtility utility) throws IOException {
    return switch (entry.kind()) {
      case SCORE_RANK -> Phi.scoreRank(entry.variant(), utility);
      case SCORE_SMOOTH -> Phi.scoreSmooth(entry.variant(), utility);
      case NETWORK -> network(entry, utility);
    };
  }

  /**
   * 値の Phi を開く。GRP のモードは {@link #open(PhiEntry, RankUtility)} と同じ。
   *
   * @param lateSlots LATE のモードで、数え上げた卓の Φ を覚えるメモのスロットの数（2 の冪。値には効かない）。メモは全てのスレッドで
   *     共有する
   * @throws IOException チェックポイントか表か q が読めないか、sha が項目と違うとき
   * @throws IllegalStateException ルールが項目と違うとき、GruSinkhorn 以外のモデルに表を重ねようとしたとき、LATE のモードの葉が
   *     GrpNetwork でないとき
   */
  public static Phi open(PhiValue value, RankUtility utility, int lateSlots) throws IOException {
    return switch (value.mode()) {
      case GRP -> open(value.entry(), utility);
      case LATE -> late(value, utility, lateSlots);
    };
  }

  /**
   * 値の Phi を、終盤のメモを 1 スロットにして開く。開けるかを確かめるだけのとき（check-config・grp-switch）と、局末ごとに 1 回ずつ
   * 求めるとき（裏の採点）に使い、共有のメモの大きな配列を確保しない。メモは値に効かないので、値は
   * {@link #open(PhiValue, RankUtility, int)} で開いた Phi とビット単位で同じ。投げる例外もそちらと同じ。
   */
  public static Phi openSingleSlot(PhiValue value, RankUtility utility) throws IOException {
    return open(value, utility, 1);
  }

  /**
   * チェックポイントの項目を作る。sha は今あるファイルから求め、チェックポイントのパスは絶対パスにして書く。
   *
   * @param table 重ねる PhiTable のファイル（重ねなければ空）
   * @param outcomes q のチェックポイントのディレクトリ（持たなければ null）。項目には絶対パスにして書く
   * @throws IOException チェックポイントか表か q が読めないとき
   */
  static PhiEntry checkpointEntry(
      String id, Variant variant, Path checkpoint, String table, Path outcomes, int effectiveFrom, String parent,
      String reason) throws IOException {
    Path exact = checkpoint.toAbsolutePath().normalize();
    String tableSha = table.isEmpty() ? "" : PhiTable.sha256(Path.of(table));
    Path exactOutcomes = outcomes == null ? null : outcomes.toAbsolutePath().normalize();
    return new PhiEntry(
        id,
        PhiEntryKind.NETWORK,
        variant,
        exact.toString(),
        GrpCheckpoint.sha256(exact),
        table,
        tableSha,
        exactOutcomes == null ? "" : exactOutcomes.toString(),
        exactOutcomes == null ? "" : GrpCheckpoint.sha256(exactOutcomes),
        effectiveFrom,
        parent,
        reason);
  }

  private static Phi network(PhiEntry entry, RankUtility utility) throws IOException {
    GrpModel model = model(entry);
    Path directory = Path.of(entry.path());
    if (!entry.table().isEmpty()) {
      Path file = Path.of(entry.table());
      String tableSha = PhiTable.sha256(file);
      if (!tableSha.equals(entry.tableSha256()))
        throw new IOException("Φ table " + file + " has sha " + tableSha + ", not " + entry.tableSha256());
      PhiTable table = PhiTable.read(file);
      if (!(model instanceof GruSinkhorn gru))
        throw new IllegalStateException("A Φ table overlays a GruSinkhorn only: " + directory);
      if (table.variant() != entry.variant())
        throw new IllegalStateException("Φ table " + file + " is " + table.variant() + ", the entry " + entry.variant());
      return Phi.of(gru, utility, table);
    }
    return Phi.of(model, utility);
  }

  /**
   * 項目の q を、ルールを照合して読む。項目が q の sha を持てば（run の外から来た q）、sha も照合する。
   *
   * @throws IOException q が読めないか、sha が項目と違うとき
   * @throws IllegalStateException 項目が q を持たないとき、q のルールが項目と違うとき
   */
  static OutcomeNetwork outcomes(PhiEntry entry) throws IOException {
    if (!entry.hasOutcomes()) throw new IllegalStateException("Φ entry " + entry.id() + " has no outcome network");
    Path directory = Path.of(entry.outcomes());
    if (!entry.outcomesSha256().isEmpty()) {
      String sha = GrpCheckpoint.sha256(directory);
      if (!sha.equals(entry.outcomesSha256()))
        throw new IOException("Outcome network " + directory + " has sha " + sha + ", not " + entry.outcomesSha256());
    }
    OutcomeNetwork outcomes = GrpCheckpoint.readOutcomes(directory);
    if (outcomes.variant() != entry.variant())
      throw new IllegalStateException(
          "Outcome network " + directory + " is " + outcomes.variant() + ", the entry " + entry.variant());
    return outcomes;
  }

  private static Phi late(PhiValue value, RankUtility utility, int lateSlots) throws IOException {
    PhiEntry entry = value.entry();
    if (!(model(entry) instanceof GrpNetwork leaf))
      throw new IllegalStateException("The late mode enumerates with a GrpNetwork leaf: " + entry.path());
    return Phi.late(FinalRound.of(outcomes(entry), leaf, value.cutoff()), utility, lateSlots);
  }

  /** 項目の GRP のチェックポイントを、ルールを照合して読む。項目が sha を持てば（run の外から来たチェックポイント）、sha も照合する。 */
  private static GrpModel model(PhiEntry entry) throws IOException {
    Path directory = Path.of(entry.path());
    if (!entry.sha256().isEmpty()) {
      String sha = GrpCheckpoint.sha256(directory);
      if (!sha.equals(entry.sha256()))
        throw new IOException("GRP checkpoint " + directory + " has sha " + sha + ", not " + entry.sha256());
    }
    GrpModel model = GrpCheckpoint.read(directory);
    if (model.variant() != entry.variant())
      throw new IllegalStateException("GRP checkpoint " + directory + " is " + model.variant() + ", the entry " + entry.variant());
    return model;
  }
}
