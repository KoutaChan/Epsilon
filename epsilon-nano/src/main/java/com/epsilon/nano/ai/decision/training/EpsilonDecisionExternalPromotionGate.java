package com.epsilon.nano.ai.decision.training;

import com.epsilon.nano.ai.decision.audit.EpsilonDecisionProductionAuditProtocol;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/** 独立した固定牌山の評価結果を検証し、対局収集用の採用モデルを本番用に採用する。 */
public final class EpsilonDecisionExternalPromotionGate {

  private static final int SEAT_ROTATIONS = 4;

  private EpsilonDecisionExternalPromotionGate() {}

  /**
   * 外部固定牌山レポートを検証し、そこに記録された対局収集に使う採用モデルを本番にする。
   *
   * <p>レポートの規約、判定、牌山数、4席を入れ替えた対局数、対応をそろえた下限、候補・比較元 IDをすべて照合する。
   *
   * @param checkpointRoot 対局・本番参照を管理するチェックポイントのルートディレクトリ
   * @param candidate 昇格対象対局収集に使う採用モデルチェックポイント
   * @param fixedWallReport 外部評価が出力したキーと値のレポート
   * @return 適用した採用と検証済み監査識別情報
   * @throws IOException レポートまたはチェックポイントの検証・更新に失敗した場合
   */
  public static PromotionResult promote(Path checkpointRoot, Path candidate, Path fixedWallReport)
      throws IOException {
    Path root = checkpointRoot.toAbsolutePath().normalize();
    Path normalizedCandidate = candidate.toAbsolutePath().normalize();
    Path report = fixedWallReport.toAbsolutePath().normalize();
    Map<String, String> fields = parseReport(Files.readString(report));
    require(fields, "protocolId", EpsilonDecisionProductionAuditProtocol.ID);
    require(fields, "decision", "PROMOTED");
    require(fields, "exactWallSeeds", "true");
    require(fields, "continuousArena", "true");
    require(fields, "confidenceMethod", "FIXED_SAMPLE_GAUSSIAN");
    int requestedWallSeeds = positiveInt(fields, "requestedWallSeeds");
    int wallSeeds = positiveInt(fields, "wallSeeds");
    int games = positiveInt(fields, "games");
    if (requestedWallSeeds != EpsilonDecisionProductionAuditProtocol.WALL_SEEDS
        || wallSeeds != EpsilonDecisionProductionAuditProtocol.WALL_SEEDS) {
      throw new IOException(
          "External fixed-wall report does not match the production audit wall budget: requested="
              + requestedWallSeeds
              + " actual="
              + wallSeeds
              + " required="
              + EpsilonDecisionProductionAuditProtocol.WALL_SEEDS);
    }
    if (games != Math.multiplyExact(wallSeeds, SEAT_ROTATIONS)) {
      throw new IOException(
          "External fixed-wall report game count does not match four seat rotations: games="
              + games
              + " wallSeeds="
              + wallSeeds);
    }
    double alpha = finiteDouble(fields, "alpha");
    double lower = finiteDouble(fields, "pairedRankDeltaLower");
    double promotionMargin = finiteDouble(fields, "promotionMargin");
    double harmfulMargin = finiteDouble(fields, "harmfulMargin");
    if (Double.compare(alpha, EpsilonDecisionProductionAuditProtocol.ALPHA) != 0
        || Double.compare(promotionMargin, EpsilonDecisionProductionAuditProtocol.PROMOTION_MARGIN)
            != 0
        || Double.compare(harmfulMargin, EpsilonDecisionProductionAuditProtocol.HARMFUL_MARGIN)
            != 0) {
      throw new IOException(
          "External fixed-wall report does not match the production audit contract:"
              + " alpha="
              + alpha
              + " promotionMargin="
              + promotionMargin
              + " harmfulMargin="
              + harmfulMargin);
    }
    if (!(lower > EpsilonDecisionProductionAuditProtocol.PROMOTION_MARGIN)) {
      throw new IOException(
          "External fixed-wall report lower bound does not clear the required delta: lower="
              + lower
              + " promotionMargin="
              + EpsilonDecisionProductionAuditProtocol.PROMOTION_MARGIN);
    }

    Path reportCandidate = normalizedPath(fields, "candidateCheckpoint");
    Path reportParent = normalizedPath(fields, "parentCheckpoint");
    if (!normalizedCandidate.equals(reportCandidate)) {
      throw new IOException(
          "External fixed-wall report does not name the exact campaign candidate");
    }
    long seedBase = longValue(fields, "seedBase");
    long duelSequence = positiveLong(fields, "duelSequence");
    String auditId = requiredField(fields, "auditId");
    String expectedAuditId = EpsilonDecisionProductionAuditProtocol.auditId(seedBase, duelSequence);
    if (!expectedAuditId.equals(auditId)) {
      throw new IOException(
          "External fixed-wall auditId does not match its seed series: expected="
              + expectedAuditId
              + " actual="
              + auditId);
    }

    EpsilonDecisionCheckpointBundle candidateManifest =
        EpsilonDecisionCheckpointManager.loadManifest(normalizedCandidate);
    String candidateId = EpsilonDecisionCheckpointManager.candidateId(normalizedCandidate);
    requireCanonicalCandidate(root, normalizedCandidate, candidateId, "candidate");
    String parentCandidateId = EpsilonDecisionCheckpointManager.candidateId(reportParent);
    requireCanonicalCandidate(root, reportParent, parentCandidateId, "parent");

    String evidenceRef =
        "protocolId="
            + EpsilonDecisionProductionAuditProtocol.ID
            + ",auditId="
            + auditId
            + ",report="
            + report
            + ",wallSeeds="
            + wallSeeds
            + ",alpha="
            + alpha
            + ",lower="
            + lower
            + ",promotionMargin="
            + promotionMargin;
    EpsilonDecisionChampionStore.promoteProductionChampion(
        root,
        normalizedCandidate,
        evidenceRef,
        parentCandidateId,
        auditId,
        EpsilonDecisionProductionAuditProtocol.ID);
    return new PromotionResult(
        normalizedCandidate,
        reportParent,
        candidateManifest.iteration,
        wallSeeds,
        games,
        lower,
        promotionMargin,
        candidateId,
        parentCandidateId,
        auditId,
        EpsilonDecisionProductionAuditProtocol.ID);
  }

  static Map<String, String> parseReport(String report) throws IOException {
    HashMap<String, String> fields = new HashMap<>();
    for (String rawLine : report.split("\\R")) {
      String line = rawLine.trim();
      if (!line.contains("=")) {
        continue;
      }
      int separator = line.indexOf('=');
      String key = line.substring(0, separator).trim();
      String value = line.substring(separator + 1).trim();
      if (key.isEmpty() || value.isEmpty()) {
        throw new IOException("Malformed external fixed-wall report line: " + rawLine);
      }
      if (fields.putIfAbsent(key, value) != null) {
        throw new IOException("Duplicate external fixed-wall report key: " + key);
      }
    }
    return Map.copyOf(fields);
  }

  private static void require(Map<String, String> fields, String key, String expected)
      throws IOException {
    String actual = fields.get(key);
    if (!expected.equals(actual)) {
      throw new IOException(
          "External fixed-wall report field mismatch: "
              + key
              + " expected="
              + expected
              + " actual="
              + actual);
    }
  }

  private static int positiveInt(Map<String, String> fields, String key) throws IOException {
    String raw = requiredField(fields, key);
    try {
      int value = Integer.parseInt(raw);
      if (value <= 0) {
        throw new IOException("External fixed-wall report field must be positive: " + key);
      }
      return value;
    } catch (NumberFormatException error) {
      throw new IOException("Invalid integer in external fixed-wall report: " + key, error);
    }
  }

  private static long positiveLong(Map<String, String> fields, String key) throws IOException {
    long value = longValue(fields, key);
    if (value <= 0L) {
      throw new IOException("External fixed-wall report field must be positive: " + key);
    }
    return value;
  }

  private static long longValue(Map<String, String> fields, String key) throws IOException {
    try {
      return Long.parseLong(requiredField(fields, key));
    } catch (NumberFormatException error) {
      throw new IOException("Invalid long in external fixed-wall report: " + key, error);
    }
  }

  private static double finiteDouble(Map<String, String> fields, String key) throws IOException {
    String raw = requiredField(fields, key);
    try {
      double value = Double.parseDouble(raw);
      if (!Double.isFinite(value)) {
        throw new IOException("External fixed-wall report field must be finite: " + key);
      }
      return value;
    } catch (NumberFormatException error) {
      throw new IOException("Invalid double in external fixed-wall report: " + key, error);
    }
  }

  private static Path normalizedPath(Map<String, String> fields, String key) throws IOException {
    try {
      return Path.of(requiredField(fields, key)).toAbsolutePath().normalize();
    } catch (RuntimeException error) {
      throw new IOException("Invalid path in external fixed-wall report: " + key, error);
    }
  }

  private static void requireCanonicalCandidate(
      Path root, Path checkpoint, String candidateId, String label) throws IOException {
    Path expected = root.resolve("candidate").resolve(candidateId);
    if (!checkpoint.equals(expected)) {
      throw new IOException(
          "External fixed-wall "
              + label
              + " is not the canonical campaign candidate: expected="
              + expected
              + " actual="
              + checkpoint);
    }
  }

  private static String requiredField(Map<String, String> fields, String key) throws IOException {
    String value = fields.get(key);
    if (value == null || value.isBlank()) {
      throw new IOException("Missing external fixed-wall report field: " + key);
    }
    return value;
  }

  /**
   * 外部固定牌山分岐を通過して適用した採用の監査結果。
   *
   * @param candidate 昇格した候補チェックポイント
   * @param parent 昇格前の本番採用モデルチェックポイント
   * @param iteration 候補反復回数
   * @param wallSeeds 評価に使った独立牌山数
   * @param games 評価した総対局数
   * @param pairedRankDeltaLower 対応をそろえた順位差の信頼下限
   * @param promotionMargin 本番採用に必要な事前固定境界
   * @param candidateId 昇格した候補の不変ID
   * @param parentCandidateId 監査時本番比較元の不変ID
   * @param auditId 独立監査を一意に識別するID
   * @param auditProtocolId 監査仕様のID
   */
  public record PromotionResult(
      Path candidate,
      Path parent,
      int iteration,
      int wallSeeds,
      int games,
      double pairedRankDeltaLower,
      double promotionMargin,
      String candidateId,
      String parentCandidateId,
      String auditId,
      String auditProtocolId) {

    /**
     * 採用の監査項目を一行一項目のレポートに変換する。
     *
     * @return 末尾改行を含むキーと値のレポート
     */
    public String render() {
      return "state=PRODUCTION_CHAMPION\n"
          + "candidate="
          + candidate
          + "\nparent="
          + parent
          + "\niteration="
          + iteration
          + "\nwallSeeds="
          + wallSeeds
          + "\ngames="
          + games
          + "\npairedRankDeltaLower="
          + pairedRankDeltaLower
          + "\npromotionMargin="
          + promotionMargin
          + "\ncandidateId="
          + candidateId
          + "\nparentCandidateId="
          + parentCandidateId
          + "\nauditId="
          + auditId
          + "\nauditProtocolId="
          + auditProtocolId
          + '\n';
    }
  }
}
