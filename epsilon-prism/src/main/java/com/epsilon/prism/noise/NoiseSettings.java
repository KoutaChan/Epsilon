package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import java.util.Objects;

/**
 * ラベルのノイズを減らす方法のセクション。どの方法も、キー 1 つで止められる。
 *
 * <p>この jar で選べるのは次のものだけ。共通の牌山、席ごとのツモ列を使うか、手を引く乱数の 2 つの共有のしかた、帳簿の 4 つのモード、
 * TEMPO を使うかとその値、判断のクラスごとの率の表、全ての合法手を測るクラス。帳簿では、計算する組と補正に使う組を
 * {@link LedgerGroup#computed} の 5 組から選べ、切る幅・リッジの強さ・区分に要る行の数・偏りを警告するしきい値も設定の値を使う。
 * 裏ドラは実際の値で学び、監査のために期待値も求める局末の割合は設定の値を使う。ほかの方法（帳簿の新しい組、係数を当てはめる区分と行、
 * 当てはめの重みと縮め先、率の倍率と見直し、裏ドラの期待値で学ぶこと、複製、間引き、クラスごとのゲイン）のキーは既定値だけを受け付け、
 * ほかの値はエラーにする。
 *
 * <p>TEMPO で {@link NoiseGain#LEDGER_R2} を使うときは、帳簿のモードを APPLY にする。裏ドラを期待値で学ぶときは、帳簿で
 * {@link LedgerGroup#URA} を計算しない。
 *
 * @param crn 共通乱数
 * @param ledger 運の精算の帳簿
 * @param lossWeight 帳簿の係数を当てはめるときの重みと縮め先
 * @param tempo TEMPO
 * @param rates 判断のクラスごとのラベル行の率
 * @param ura 立直の和了で終わった局の、局末の値の作り方（裏ドラの扱い）
 * @param replicas 複製の行
 * @param thinning 間引き
 * @param classGain 判断のクラスごとのゲイン
 */
public record NoiseSettings(
    @Key(doc = "共通乱数") CrnSettings crn,
    @Key(doc = "運の精算の帳簿のモード", value = "APPLY") LedgerSettings ledger,
    @Key(doc = "帳簿の係数を当てはめるときの重みと、係数の縮め先", change = Change.FIT) LossWeightSettings lossWeight,
    @Key(doc = "TEMPO を使うか", value = "true", change = Change.FIT) TempoSettings tempo,
    @Key(doc = "判断のクラスごとのラベル行の率") RateSettings rates,
    @Key(doc = "立直の和了で終わった局の、局末の値の作り方", value = "REALIZED") UraSettings ura,
    @Key(doc = "複製の行を作るか", value = "false") ReplicaSettings replicas,
    @Key(doc = "間引きをするか", value = "false") ThinningSettings thinning,
    @Key(doc = "判断のクラスごとのゲインのモード", value = "OFF", change = Change.FIT) ClassGainSettings classGain) {

  public NoiseSettings {
    Objects.requireNonNull(crn, "crn");
    Objects.requireNonNull(ledger, "ledger");
    Objects.requireNonNull(lossWeight, "lossWeight");
    Objects.requireNonNull(tempo, "tempo");
    Objects.requireNonNull(rates, "rates");
    Objects.requireNonNull(ura, "ura");
    Objects.requireNonNull(replicas, "replicas");
    Objects.requireNonNull(thinning, "thinning");
    Objects.requireNonNull(classGain, "classGain");
    if (tempo instanceof TempoSettings.On on && on.noiseGain() == NoiseGain.LEDGER_R2 && !ledger.corrects()) {
      throw new InvalidSetting(
          "tempo.noiseGain", "noiseGain LEDGER_R2 raises K′ by the R² of a correction, so it needs ledger mode APPLY");
    }
    if (ura instanceof UraSettings.Expected
        && ledger instanceof LedgerSettings.Recording recording
        && recording.compute().contains(LedgerGroup.URA)) {
      throw new InvalidSetting(
          "ledger.compute",
          "the expected ura dora already averages the ura indicators, so the ledger must not compute URA");
    }
    requireAvailable(crn, ledger, lossWeight, tempo, rates, ura, replicas, thinning, classGain);
  }

  /** この jar の収集と学習が扱える値だけを受け付け、ほかの値はエラーにする。 */
  private static void requireAvailable(
      CrnSettings crn,
      LedgerSettings ledger,
      LossWeightSettings lossWeight,
      TempoSettings tempo,
      RateSettings rates,
      UraSettings ura,
      ReplicaSettings replicas,
      ThinningSettings thinning,
      ClassGainSettings classGain) {
    if (crn.wall() != WallSharing.SHARED) {
      throw unavailable("crn.wall", "wall " + crn.wall());
    }
    if (crn.actions() == ActionSharing.INDEPENDENT) {
      throw unavailable("crn.actions", "actions " + crn.actions());
    }
    if (ledger instanceof LedgerSettings.Recording recording
        && !LedgerGroup.computed().containsAll(recording.compute())) {
      throw unavailable("ledger.compute", "a ledger that computes groups outside " + LedgerGroup.computed());
    }
    if (ledger instanceof LedgerSettings.Fitting fitting
        && (fitting.cells() != LedgerCells.CLASS || fitting.coefficients() != CoefficientSource.OTHER_PARITY)) {
      throw unavailable("ledger", "a ledger fit other than cells CLASS and coefficients OTHER_PARITY");
    }
    if (lossWeight.fit() != LossWeightSettings.Weighting.UNIFORM || lossWeight.shrink() != LossWeightSettings.Shrink.ZERO) {
      throw unavailable("lossWeight", "a policy weight or a shrink target");
    }
    if (tempo instanceof TempoSettings.On on && on.noiseGain() == NoiseGain.EFFECTIVE) {
      throw unavailable("tempo.noiseGain", "noiseGain EFFECTIVE");
    }
    if (rates.multiplier() != 1) {
      throw unavailable("rates.multiplier", "a rate multiplier other than 1");
    }
    if (!(rates.review() instanceof RateReview.Off)) {
      throw unavailable("rates.review", "a rate review");
    }
    if (ura instanceof UraSettings.Expected) {
      throw unavailable("ura", "the expected ura dora");
    }
    if (!(replicas instanceof ReplicaSettings.Off)) {
      throw unavailable("replicas", "replicas");
    }
    if (!(thinning instanceof ThinningSettings.Off)) {
      throw unavailable("thinning", "thinning");
    }
    if (!(classGain instanceof ClassGainSettings.Off)) {
      throw unavailable("classGain", "class gains");
    }
  }

  private static InvalidSetting unavailable(String component, String what) {
    return new InvalidSetting(component, what + " is not available in this jar");
  }
}
