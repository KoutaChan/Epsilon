package com.epsilon.reviewer.api;

import com.epsilon.reviewer.store.OwnerId;

/** 要求した人。デスクトップでは常に {@code Owner(OwnerId.LOCAL)} になる。 */
public sealed interface Caller {
  record Owner(OwnerId owner) implements Caller {}

  /** 公開のパスと共有リンクの閲覧者（オーナーの権限は持たない）。 */
  record Visitor() implements Caller {}
}
