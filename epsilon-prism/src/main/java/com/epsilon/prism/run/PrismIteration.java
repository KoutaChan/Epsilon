package com.epsilon.prism.run;

import com.epsilon.agent.run.IterationFile;
import com.epsilon.agent.run.Iteration;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.prism.grp.ActivePhi;
import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.luck.LedgerFit;
import com.epsilon.prism.luck.LuckSettlement;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * 系譜の 1 つの反復。反復 t はバージョン {@link #start} で集め、そのネットワークから生徒の学習を始めて、バージョン t + 1 を公開する。
 * 収集と学習の設定のキーは、この反復に固定した設定の行から読む。デバイス・スレッド・バッチ・推論のバックエンドは、JVM を起動したときの
 * 行から読む。
 */
public final class PrismIteration implements Iteration {

  private final IterationFile fixed;
  private final PrismSettings settings;
  private final PrismState state;
  private final PrismSession session;

  /** 反復の運の精算（反復に固定した設定の noise と、四人打ち・三人打ちのルールから作る）。 */
  private final LuckSettlement luck;

  /** 確定したコホートを打ったバージョン（収集の段が完了マーカーの報告から読む。まだなら null）。 */
  private Integer collected;

  /** Φ の段が決めた Φ か、完了マーカーから開いた Φ（まだなら null）。 */
  private ActivePhi phi;

  /** 収集のライターが作った帳簿の和（同じ JVM で集めたときだけ）。 */
  private LedgerFit.WriterTap ledger;

  PrismIteration(IterationFile fixed, PrismSettings settings, PrismState state, PrismSession session) {
    this.fixed = Objects.requireNonNull(fixed, "fixed");
    this.settings = Objects.requireNonNull(settings, "settings");
    this.state = Objects.requireNonNull(state, "state");
    this.session = Objects.requireNonNull(session, "session");
    luck = LuckSettlement.of(settings.noise(), settings.run().variant());
  }

  @Override
  public int number() {
    return fixed.iteration();
  }

  @Override
  public Path work() {
    return session.run().work(number());
  }

  @Override
  public int settingsRow() {
    return fixed.settingsRow();
  }

  @Override
  public String jar() {
    return fixed.jar();
  }

  @Override
  public int launchRow() {
    return session.launchRow();
  }

  /** この反復に固定した行の設定。 */
  public PrismSettings settings() {
    return settings;
  }

  /** JVM を起動したときの行の設定（デバイス・スレッド・バッチ・推論のバックエンドに使う）。 */
  public PrismSettings machine() {
    return session.machine();
  }

  /** 反復を始めたときの state。 */
  public PrismState state() {
    return state;
  }

  /**
   * 全席が打ち、生徒の学習を始めるバージョン。コホートを確定した反復では、そのコホートを打ったバージョンを返す（確定した後に state の
   * 始めのバージョンが替わっても、集めたバージョンから学習して公開する）。
   */
  public int start() {
    return collected != null ? collected : state.start();
  }

  /** 収集の段が、確定したコホートを打ったバージョンを渡す。 */
  void collected(int start) {
    collected = start;
  }

  /** 反復が公開するバージョン。 */
  public int version() {
    return number() + 1;
  }

  RunDirectory run() {
    return session.run();
  }

  PrismSession session() {
    return session;
  }

  /**
   * 反復の Φ。Φ の段が同じ JVM で決めていなければ、完了マーカーに書いた値から開く。
   *
   * @throws IOException 完了マーカーが読めないか、値の GRP か q が開けないとき
   */
  ActivePhi phi() throws IOException {
    if (phi == null) {
      phi = session.resources().phi(PhiStep.recorded(this).value());
    }
    return phi;
  }

  void phi(ActivePhi phi) {
    this.phi = phi;
  }

  /**
   * 反復の Φ が使う GRP の項目。Φ を開かずに、決めた Φ か完了マーカーから読む。
   *
   * @throws IOException 完了マーカーが読めないとき
   */
  PhiEntry phiEntry() throws IOException {
    return phi != null ? phi.entry() : PhiStep.recorded(this).value().entry();
  }

  /** 反復の運の精算。 */
  LuckSettlement luck() {
    return luck;
  }

  /** 収集のライターが作った帳簿の和。同じ JVM で集めていなければ null。 */
  LedgerFit.WriterTap ledger() {
    return ledger;
  }

  void ledger(LedgerFit.WriterTap ledger) {
    this.ledger = ledger;
  }
}
