package com.epsilon.prism.grp;

/** 関門に合格した候補の入れ方（設定 {@code epsilon.prism.grpSwitch}）。 */
public enum GrpSwitch {

  /** 候補を refits.jsonl に READY で公開するだけで、入れるのは手動の grp-switch。 */
  PROPOSE,

  /** 関門に合格した候補を、READY になった後に始まる最初の反復から入れる（{@link AutoSwitch}）。 */
  AUTO
}
