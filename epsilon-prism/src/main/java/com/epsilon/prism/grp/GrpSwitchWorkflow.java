package com.epsilon.prism.grp;

import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Φ の手動の切り替えと巻き戻し（grp-switch）。--to は READY の候補の試行 id か、既にある項目の id（巻き戻し。例 g000）で、--from の反復から
 * 効く項目を phi.jsonl に 1 行足す（{@link PhiHistory#append}。始まった反復以前は拒む）。--from を省くと、始まった最後の反復の次（学習の JVM が
 * 動いていればその次。{@link PhiHistory#firstOpenIteration}）。phiSource = SELF_GRP の run だけ。
 *
 * <p>足す前に確かめること: 候補は refits.jsonl の最後の行が READY（境目の点検で REJECTED になった候補は入れ直さない）、GRP の項目は checkpoint
 * の sha が今も同じ、新しい候補は境目の点検の見本がある。その反復に既に同じ Φ が効いていれば何もしない（状態 1）。
 *
 * <p>本番の jar の Main で打つので、走らせ役の照合（学習の JVM の待ち）に掛かる。反復の境目を避け、GPU を隠して（{@code PRISM_GPU=}）打つ。
 */
public final class GrpSwitchWorkflow implements CommandWorkflow<Integer> {
  private static final Logger LOG = LoggerFactory.getLogger(GrpSwitchWorkflow.class);

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "grp-switch",
        WorkflowKind.PROMOTION,
        "grp-switch --run DIR --to ID [--from T] [--reason TEXT]",
        4,
        8);
  }

  @Override
  public Integer execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    String to = options.optionalText("to", null);
    String fromText = options.optionalText("from", null);
    String reason = options.optionalText("reason", "grp-switch");
    options.requireEmpty();
    if (to == null) throw new IllegalArgumentException("grp-switch requires --to");
    GrpRefitWorkflow.selfGrpConfig(run);
    var files = new GrpFiles(run);
    List<PhiEntry> entries = PhiHistory.read(files);
    if (entries.isEmpty())
      throw new IllegalStateException("No g000 in " + files.phiHistory() + ": start the SELF_GRP lineage first");
    int from = fromText == null ? PhiHistory.firstOpenIteration(files) : Integer.parseInt(fromText);
    PhiEntry entry = target(files, entries, to, from, reason);
    Optional<PhiEntry> effective = PhiHistory.effectiveAt(entries, from);
    if (effective.isPresent() && effective.get().sameValue(entry)) {
      LOG.warn("grp-switch: {} is already the Φ at iteration {} ({}); nothing appended", to, from, effective.get().id());
      return 1;
    }
    PhiHistory.append(files, entry);
    LOG.info(
        "grp-switch: Φ {} from iteration {} (was {}): {}",
        entry.id(),
        from,
        effective.map(PhiEntry::id).orElse("-"),
        entry);
    return 0;
  }

  @Override
  public String summarize(Integer status) {
    return "status=" + status;
  }

  @Override
  public boolean successful(Integer status) {
    return status == 0;
  }

  /** --to の項目（既にある項目なら巻き戻し、なければ READY の候補）を反復 from から効かせる行。 */
  private static PhiEntry target(GrpFiles files, List<PhiEntry> entries, String to, int from, String reason)
      throws IOException {
    Optional<RefitRecord> record = RefitRecord.latest(RefitRecord.readAll(files), to);
    if (record.isPresent() && record.get().status() != RefitStatus.READY)
      throw new IllegalArgumentException("Candidate " + to + " is " + record.get().status() + ": " + record.get().detail());
    PhiEntry existing = null;
    for (PhiEntry entry : entries) if (entry.id().equals(to)) existing = entry;
    PhiEntry entry;
    if (existing != null) {
      entry = existing.from(from, "rollback: " + reason);
    } else {
      if (record.isEmpty()) throw new IllegalArgumentException("No Φ entry or refit attempt named " + to);
      entry = record.get().entry(from, reason);
      if (!Files.exists(files.boundary(to)))
        throw new IllegalStateException("Candidate " + to + " has no boundary check " + files.boundary(to));
    }
    if (!entry.kind().scoreBased()) {
      String sha = EpsilonGrpCheckpointManager.checkpointSha256(Path.of(entry.path()));
      if (!sha.equals(entry.sha256()))
        throw new IllegalStateException("GRP checkpoint " + entry.path() + " has sha " + sha + ", not " + entry.sha256());
    }
    return entry;
  }
}
