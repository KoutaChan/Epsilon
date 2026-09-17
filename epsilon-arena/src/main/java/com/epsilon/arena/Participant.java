package com.epsilon.arena;

import com.epsilon.spi.BatchedPolicy;

/** 対局の参加者。推論の実行環境は複数の席で共有でき、呼び出し元が作成と解放を担当する。 */
public record Participant(String name, BatchedPolicy policy) {}
