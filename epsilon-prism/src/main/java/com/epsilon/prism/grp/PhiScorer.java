package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 裏の採点（採点の錠を持って呼ぶ）: 対局の記録があり phimon.jsonl にまだないコホートを、記録の先頭の Φ の項目（{@link HandEndPhi#of}）と影の
 * Φ（READY の候補、WITH_ORIGINAL なら元の GRP）で採点し（{@link CellStats#measure}）、EWMA を進めて 1 行書く。
 *
 * <p>使用中の項目は記録の先頭の id で引き直し（{@value PhiEntry#FIXED_ID} は grpCheckpoint の今の子、点数だけの Φ は固定の項目、それ以外は
 * phi.jsonl の同じ id の行）、sha が記録と違えばそのコホートを採点しない。影の候補は phiSource = SELF_GRP のときだけ、refits.jsonl の最後の
 * 状態が READY で使用中でない新しい {@value #SHADOW_CANDIDATES} つ。開けない影はその影だけを飛ばす。
 */
public final class PhiScorer {
  private static final Logger LOG = LoggerFactory.getLogger(PhiScorer.class);

  /** 採点を飛ばすコホートの遅れ（最新の記録よりこれだけ古い未採点のコホートは採点しない）。 */
  public static final int MAX_LAG = 8;

  /** 影として採点する READY の候補の数の上限（新しい順。1 つの GRP の採点は CPU で数十秒）。 */
  static final int SHADOW_CANDIDATES = 2;

  private static final Pattern GAME_LOG = Pattern.compile("c(\\d{5})\\.bin\\.gz");

  /** 採点の道具（試験は差し替える）。 */
  interface Tools {
    /** 記録の先頭の項目を引き直す。 */
    PhiEntry active(GrpFiles files, PrismConfig config, GameLogHeader header) throws IOException;

    /** 影の項目（使用中の項目を除く）。 */
    List<PhiEntry> shadows(GrpFiles files, PrismConfig config, PhiEntry active) throws IOException;

    /** 項目の Φ を開く。 */
    HandEndPhi open(PhiEntry entry, EpsilonUtilityProfile profile) throws IOException;

    /** 1 つの Φ の較正の測り。 */
    CellReport measure(List<RecordedGame> games, PhiValues values, EpsilonUtilityProfile profile);
  }

  /** 本番の道具。 */
  static final Tools PRODUCTION =
      new Tools() {
        @Override
        public PhiEntry active(GrpFiles files, PrismConfig config, GameLogHeader header) throws IOException {
          PhiEntry entry =
              switch (header.phiId()) {
                case PhiEntry.FIXED_ID -> original(config);
                case PhiEntry.SCORE_RANK_ID -> PhiEntry.scoreRank();
                case PhiEntry.SCORE_SMOOTH_ID -> PhiEntry.scoreSmooth();
                default -> {
                  PhiEntry found = null;
                  for (PhiEntry candidate : PhiHistory.read(files))
                    if (candidate.id().equals(header.phiId())) found = candidate;
                  if (found == null)
                    throw new IOException("Φ entry " + header.phiId() + " is not in " + files.phiHistory());
                  yield found;
                }
              };
          if (!entry.sha256().equals(header.phiSha256()))
            throw new IOException(
                "Φ entry " + entry.id() + " is now " + entry.sha256() + ", the games were valued with "
                    + header.phiSha256());
          return entry;
        }

        @Override
        public List<PhiEntry> shadows(GrpFiles files, PrismConfig config, PhiEntry active) throws IOException {
          List<PhiEntry> shadows = new ArrayList<>();
          if (config.phiSource() == PhiSource.SELF_GRP) {
            // 同じ id の後の行が前の行に勝つ。READY になった順に並べ、新しいものを取る
            Map<String, RefitRecord> latest = new LinkedHashMap<>();
            for (RefitRecord record : RefitRecord.readAll(files)) {
              latest.remove(record.id());
              latest.put(record.id(), record);
            }
            List<RefitRecord> ready = new ArrayList<>();
            for (RefitRecord record : latest.values())
              if (record.status() == RefitStatus.READY && !record.id().equals(active.id())) ready.add(record);
            for (RefitRecord record : ready.subList(Math.max(0, ready.size() - SHADOW_CANDIDATES), ready.size()))
              shadows.add(record.entry(active.effectiveFrom(), "shadow"));
          }
          if (config.phiShadowReference() == PhiShadowReference.WITH_ORIGINAL
              && active.kind() != PhiEntryKind.FIXED_CHECKPOINT)
            shadows.add(original(config));
          return shadows;
        }

        @Override
        public HandEndPhi open(PhiEntry entry, EpsilonUtilityProfile profile) throws IOException {
          return HandEndPhi.of(entry, profile);
        }

        @Override
        public CellReport measure(List<RecordedGame> games, PhiValues values, EpsilonUtilityProfile profile) {
          return CellStats.measure(games, values, profile);
        }
      };

  private PhiScorer() {}

  /** 採点するコホート（記録があり未採点で、遅れが {@value #MAX_LAG} 以内。昇順）。 */
  public static List<Integer> pending(GrpFiles files) throws IOException {
    TreeSet<Integer> recorded = recorded(files);
    List<Integer> pending = new ArrayList<>();
    if (recorded.isEmpty()) return pending;
    Set<Integer> scored = new HashSet<>();
    for (PhiMonitorLine line : PhiMonitorLine.readAll(files)) scored.add(line.cohort());
    int newest = recorded.last();
    for (int cohort : recorded) if (newest - cohort <= MAX_LAG && !scored.contains(cohort)) pending.add(cohort);
    return pending;
  }

  /** 対局の記録があるコホート（書き終えた grp/games/c&lt;t&gt;.bin.gz。昇順）。 */
  static TreeSet<Integer> recorded(GrpFiles files) throws IOException {
    TreeSet<Integer> recorded = new TreeSet<>();
    Path games = files.games(0).getParent();
    if (!Files.isDirectory(games)) return recorded;
    try (DirectoryStream<Path> listed = Files.newDirectoryStream(games, "c*.bin.gz")) {
      for (Path file : listed) {
        Matcher matcher = GAME_LOG.matcher(file.getFileName().toString());
        if (matcher.matches()) recorded.add(Integer.parseInt(matcher.group(1)));
      }
    }
    return recorded;
  }

  /**
   * コホート t を採点して phimon.jsonl に書く。
   *
   * @param jarSha256 作業が動いている jar の sha
   * @return 書いた行（記録が読めなければ空。1 回だけ書き残す）
   * @throws IOException phimon.jsonl が書けないとき
   */
  public static Optional<PhiMonitorLine> score(
      GrpFiles files, PrismConfig config, int cohort, String jarSha256) throws IOException {
    return score(files, config, cohort, jarSha256, PRODUCTION);
  }

  /** {@link #score(GrpFiles, PrismConfig, int, String)} を道具 tools で行う。 */
  static Optional<PhiMonitorLine> score(
      GrpFiles files, PrismConfig config, int cohort, String jarSha256, Tools tools) throws IOException {
    long started = System.nanoTime();
    var profile = EpsilonUtilityProfile.valueOf(config.utility());
    CohortGames recorded;
    PhiEntry active;
    CellReport activeReport;
    try {
      recorded = GameLog.read(files.games(cohort));
      if (recorded.header().cohort() != cohort)
        throw new IOException("The game log of cohort " + cohort + " says " + recorded.header().cohort());
      active = tools.active(files, config, recorded.header());
      activeReport = measure(tools, active, recorded.games(), profile);
    } catch (IOException | RuntimeException failure) {
      LOG.warn("c{} was not scored: {}", cohort, failure.toString());
      return Optional.empty();
    }
    Map<String, CellReport> shadows = new LinkedHashMap<>();
    List<PhiEntry> shadowEntries;
    try {
      shadowEntries = tools.shadows(files, config, active);
    } catch (IOException | RuntimeException failure) {
      LOG.warn("c{} has no shadow Φ: {}", cohort, failure.toString());
      shadowEntries = List.of();
    }
    for (PhiEntry shadow : shadowEntries) {
      try {
        shadows.put(shadow.id(), measure(tools, shadow, recorded.games(), profile));
      } catch (IOException | RuntimeException failure) {
        LOG.warn("c{} skipped the shadow Φ {}: {}", cohort, shadow.id(), failure.toString());
      }
    }
    // EWMA: 前の行が同じ項目ならそこから進め、項目が変わったら初めから
    PhiMonitorLine previous = null;
    for (PhiMonitorLine line : PhiMonitorLine.readAll(files)) if (line.cohort() < cohort) previous = line;
    boolean continues =
        previous != null
            && previous.phiId().equals(active.id())
            && previous.phiSha256().equals(active.sha256())
            && previous.ewmaCohorts() > 0
            && Double.isFinite(previous.dSpgEwma());
    double dSpg = activeReport.dSpg();
    double ewma;
    int count;
    if (!Double.isFinite(dSpg)) {
      ewma = continues ? previous.dSpgEwma() : Double.NaN;
      count = continues ? previous.ewmaCohorts() : 0;
    } else if (continues) {
      ewma = previous.dSpgEwma() + PhiMonitorLine.EWMA_WEIGHT * (dSpg - previous.dSpgEwma());
      count = previous.ewmaCohorts() + 1;
    } else {
      ewma = dSpg;
      count = 1;
    }
    var line =
        new PhiMonitorLine(
            cohort,
            active.id(),
            active.sha256(),
            jarSha256,
            activeReport,
            shadows,
            ewma,
            count,
            (System.nanoTime() - started) / 1e9);
    line.append(files);
    return Optional.of(line);
  }

  /** 1 つの Φ で記録の半荘を評価して測る。 */
  private static CellReport measure(
      Tools tools, PhiEntry entry, List<RecordedGame> games, EpsilonUtilityProfile profile) throws IOException {
    try (HandEndPhi phi = tools.open(entry, profile)) {
      return tools.measure(games, phi.gameValues(games), profile);
    }
  }

  /**
   * 元の GRP の項目（grpCheckpoint の今の子と sha。学習の JVM の {@code TrainingResources} と同じ解決）。
   *
   * @throws IOException grpCheckpoint がないか読めないとき
   */
  static PhiEntry original(PrismConfig config) throws IOException {
    if (config.grpCheckpoint().isEmpty()) throw new IOException("No grpCheckpoint for the original GRP");
    Path exact = EpsilonGrpCheckpointManager.resolveExisting(Path.of(config.grpCheckpoint()).resolve("grp"));
    if (exact == null) throw new IOException("No GRP checkpoint under " + config.grpCheckpoint());
    return PhiEntry.fixed(config, exact.toString(), EpsilonGrpCheckpointManager.checkpointSha256(exact));
  }
}
