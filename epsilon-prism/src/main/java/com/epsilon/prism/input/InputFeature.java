package com.epsilon.prism.input;

import com.epsilon.agent.observe.codec.SchemaDigest;

/** 基本の入力に加える列と、公開履歴の読み方。 */
public enum InputFeature {
  /** 各遷移の基本キーに続けて、牌の状態と公開情報の組の符号を書く。 */
  CANDIDATE_SEGMENTS {
    @Override
    void describe(SchemaDigest digest, InputFormat format) {
      for (var slot : format.candidateCodes().slots()) {
        digest.with(
            slot.parameter(),
            slot.column() + "/" + slot.firstRow() + "/" + slot.space().describe());
      }
    }
  },
  /** 物理牌と公開順を使い、イベントの各列を公開時点で確定させる。 */
  PUBLIC_ORDER_EVENTS {
    @Override
    void describe(SchemaDigest digest, InputFormat format) {
      digest.with(
          name(),
          "physical tiles, public position/"
              + format.eventCapacity()
              + ", caller on open meld, owner history through event");
    }
  };

  abstract void describe(SchemaDigest digest, InputFormat format);
}
