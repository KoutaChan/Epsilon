package com.epsilon.reviewer.api;

import com.epsilon.reviewer.store.OwnerId;

/** 要求した人。デスクトップは Owner(OwnerId.LOCAL)。 */
public sealed interface Caller {
  record Owner(OwnerId owner) implements Caller {}

  /** 公開の道と共有リンクの閲覧者（持ち主の権利を持たない）。 */
  record Visitor() implements Caller {}
}
