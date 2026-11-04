package com.epsilon.reviewer;

import com.epsilon.reviewer.store.ResultId;
import java.util.function.Supplier;

/**
 * 保存した解析結果の文書。持ち主と有無は作る時に索引で確かめ、中身は送る時に読む（HTTP の 304 では読まない）。
 *
 * @param gzip 保存したままの gzip を読む（HTTP は Content-Encoding: gzip、stdio は枠の本文で送る）。読む前に消されて
 *     いれば ReviewerException
 */
public record StoredDocument(ResultId id, Supplier<byte[]> gzip) {}
