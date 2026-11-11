package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.run.IterationFile;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.classic.ClassicHeader;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.learn.SelfPlayLearner;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * campaign の 1 回の macro。macro t は学習器のバージョン t で自己対局を打ち、そのバージョンから学んでバージョン t + 1 を公開する。収集と
 * 学習のキーは反復に固定した設定の行から、デバイス・バッチ・卓の大きさは JVM を起動したときの行から読む。決闘の段と昇格の段には、公開した
 * バージョン t + 1 を候補として渡す。
 */
public final class CampaignIteration implements DuelMacro {

  private final IterationFile fixed;
  private final ClassicSettings settings;
  private final CampaignState state;
  private final CampaignSession session;

  CampaignIteration(IterationFile fixed, ClassicSettings settings, CampaignState state, CampaignSession session) {
    this.fixed = Objects.requireNonNull(fixed, "fixed");
    this.settings = Objects.requireNonNull(settings, "settings");
    this.state = Objects.requireNonNull(state, "state");
    this.session = Objects.requireNonNull(session, "session");
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

  /** 反復に固定した設定の行の設定。 */
  public ClassicSettings settings() {
    return settings;
  }

  /** JVM を起動したときの行の設定（デバイス・バッチ・卓の大きさ）。 */
  public ClassicSettings machine() {
    return session.machine();
  }

  /** 学習器の設定。学び方は反復に固定した行から、バッチの大きさは JVM を起動したときの行から読む。 */
  public SelfPlayLearner.Settings learning() {
    SelfPlayLearner.Settings fixedRow = settings.learning();
    return new SelfPlayLearner.Settings(
        fixedRow.ppo(), fixedRow.actorRate(), fixedRow.valueRate(), fixedRow.vtrace(), machine().batches());
  }

  /** macro を始めたときの state。 */
  public CampaignState state() {
    return state;
  }

  /** 学習器が打ち、学び始めるバージョン。 */
  public int version() {
    return number();
  }

  /** macro が公開するバージョン。 */
  @Override
  public int published() {
    return number() + 1;
  }

  @Override
  public RunDirectory run() {
    return session.run();
  }

  @Override
  public EvaluationSettings evaluation() {
    return settings.evaluation();
  }

  @Override
  public Variant variant() {
    return settings.run().variant();
  }

  /** 学習器のネットワークの効用（学び始めたバージョンのチェックポイントのヘッダーから読む）。 */
  @Override
  public RankUtility utility() throws IOException {
    return network().utility();
  }

  @Override
  public long seed() {
    return settings.run().seed();
  }

  /** 学習器のネットワークの系列（学び始めたバージョンのチェックポイントのヘッダーから読む）。 */
  @Override
  public Series series() throws IOException {
    return network().architecture().series();
  }

  @Override
  public TableLoop table() {
    return session.table();
  }

  @Override
  public RunControl control() {
    return session.control();
  }

  @Override
  public PolicyOptions options() {
    return SelfPlay.options(machine(), settings);
  }

  @Override
  public NetworkRuntime runtime() {
    return session.runtime();
  }

  /** 公開したバージョンのチェックポイントを写す。 */
  @Override
  public void writeLearner(Path directory) throws IOException {
    CampaignSession.copyNetwork(run().snapshot(published()), directory, variant());
  }

  CampaignSession session() {
    return session;
  }

  private ClassicHeader network() throws IOException {
    return ClassicCheckpoint.header(run().snapshot(version()));
  }
}
