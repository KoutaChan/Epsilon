package com.epsilon.reviewer;

import com.epsilon.reviewer.store.ResultId;
import java.util.function.Supplier;

/**
 * 保存した解析結果の文書。作るときに、オーナーが合うかと結果があるかをインデックスで確かめる。中身は送るときに読む（HTTP で 304 を返すときは
 * 読まない）。
 *
 * @param gzip 保存した gzip をそのまま読む関数。HTTP では Content-Encoding: gzip を付けて、stdio ではフレームの本文として送る。読む前に結果が
 *     消されていれば ReviewerException を投げる
 */
public record StoredDocument(ResultId id, Supplier<byte[]> gzip) {}
