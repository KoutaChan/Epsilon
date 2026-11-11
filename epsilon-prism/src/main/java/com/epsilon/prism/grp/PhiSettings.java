package com.epsilon.prism.grp;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * 局末の値 Φ の出どころを決めるセクション。自分の GRP の運用のキーは、自分の GRP の設定の中にだけある。出どころと g000 は run の途中で
 * 変えられない。GRP の裏の処理は学習の JVM を起動したときの設定で始まるので、運用のキーを変えても効くのは次の起動から。固定の GRP の
 * ファイルを変えると、次に始まる反復の Φ から効く。
 */
@Choice(key = "source", doc = "Φ の出どころ")
public sealed interface PhiSettings {

  /** Φ の出どころ。 */
  PhiSource source();

  /** Φ が点数だけで決まり、GRP も表も読まないなら true。自分の GRP では、g000 が点数だけの Φ のとき true。 */
  boolean scoreOnly();

  /** 終盤の数え上げ（LATE のモード）の設定。GRP を使う出どころ（自分の GRP と固定の GRP）だけが持ち、点数だけの Φ では空になる。 */
  Optional<LateSettings> lateSettings();

  /**
   * 自分の GRP。g000 から始め、自己対局の記録で学び直したネットワークに、反復の区切りで切り替える。
   *
   * @param start g000 の Φ
   * @param grp 自分の GRP の運用の設定
   * @param late 終盤の数え上げ（LATE のモード）の設定
   */
  @Case("SELF_GRP")
  record SelfGrp(
      @Key(doc = "最初の項目の Φ", value = "SCORE_RANK", change = Change.FIXED) PhiStartSettings start,
      @Key(doc = "自分の GRP の運用の設定", change = Change.MACHINE) GrpOperations grp,
      @Key(doc = "終盤の数え上げ（LATE のモード）の設定") LateSettings late)
      implements PhiSettings {

    public SelfGrp {
      Objects.requireNonNull(start, "start");
      Objects.requireNonNull(grp, "grp");
      Objects.requireNonNull(late, "late");
      if (grp.shadow() == PhiShadowReference.WITH_ORIGINAL && !(start instanceof PhiStartSettings.GrpCheckpoint)) {
        throw new InvalidSetting(
            "grp.shadow", "shadow WITH_ORIGINAL scores the original GRP, so it needs start kind GRP_CHECKPOINT");
      }
    }

    @Override
    public PhiSource source() {
      return PhiSource.SELF_GRP;
    }

    @Override
    public boolean scoreOnly() {
      return start.kind().scoreBased();
    }

    @Override
    public Optional<LateSettings> lateSettings() {
      return Optional.of(late);
    }
  }

  /**
   * 固定の GRP。学び直さない。LATE のモードを使い始める設定なら局の結果のネットワークが要る。LATE のモードは GrpNetwork を葉にして数え上げ、表を
   * 重ねられるのは GruSinkhorn だけなので、このネットワークと表は一緒に使えない。
   *
   * @param grp GRP のチェックポイント（grp.json と weights.f32 を直接含むディレクトリ）
   * @param table GRP が GruSinkhorn のときに重ねる PhiTable のファイル。無ければ重ねない
   * @param outcomes LATE のモードで最後の局を数え上げる、局の結果のネットワークのチェックポイント（grp.json と weights.f32 を直接含むディレクトリ）。無ければ
   *     LATE のモードにならない
   * @param late 終盤の数え上げ（LATE のモード）の設定
   */
  @Case("FIXED_GRP")
  record FixedGrp(
      @Key(doc = "GRP のチェックポイント。grp.json と weights.f32 を直接含むディレクトリ", presence = Presence.REQUIRED, change = Change.FIT)
          Path grp,
      @Key(doc = "GRP が GruSinkhorn のときに重ねる PhiTable のファイル。無ければ重ねない", presence = Presence.ABSENT, change = Change.FIT)
          Path table,
      @Key(doc = "LATE のモードで最後の局を数え上げる、局の結果のネットワークのチェックポイント。grp.json と weights.f32 を直接含むディレクトリ。表とは一緒に使えない",
              presence = Presence.ABSENT, change = Change.FIT)
          Path outcomes,
      @Key(doc = "終盤の数え上げ（LATE のモード）の設定") LateSettings late)
      implements PhiSettings {

    public FixedGrp {
      Objects.requireNonNull(grp, "grp");
      Objects.requireNonNull(late, "late");
      if (outcomes != null && table != null) {
        throw new InvalidSetting(
            "outcomes", "the late mode enumerates with a GrpNetwork leaf, and a Φ table overlays a GruSinkhorn only");
      }
      if (late.from() > 0 && outcomes == null) {
        throw new InvalidSetting("late.from", "the late mode needs the outcome network: set outcomes");
      }
    }

    @Override
    public PhiSource source() {
      return PhiSource.FIXED_GRP;
    }

    @Override
    public boolean scoreOnly() {
      return false;
    }

    @Override
    public Optional<LateSettings> lateSettings() {
      return Optional.of(late);
    }
  }

  /** 局末の点数の順位を、そのまま最終順位とみなす固定の Φ。 */
  @Case("SCORE_RANK")
  record ScoreRank() implements PhiSettings {

    @Override
    public PhiSource source() {
      return PhiSource.SCORE_RANK;
    }

    @Override
    public boolean scoreOnly() {
      return true;
    }

    @Override
    public Optional<LateSettings> lateSettings() {
      return Optional.empty();
    }
  }

  /** 局末の点数のなめらかな関数で表す固定の Φ。 */
  @Case("SCORE_SMOOTH")
  record ScoreSmooth() implements PhiSettings {

    @Override
    public PhiSource source() {
      return PhiSource.SCORE_SMOOTH;
    }

    @Override
    public boolean scoreOnly() {
      return true;
    }

    @Override
    public Optional<LateSettings> lateSettings() {
      return Optional.empty();
    }
  }
}
