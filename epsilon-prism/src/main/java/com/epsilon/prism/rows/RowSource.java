package com.epsilon.prism.rows;

import java.io.IOException;
import java.util.function.Predicate;

/** 封印済みのコホートの行の読み出し口。学習・認証・保留の指標はこれだけを通して行を読む。 */
public interface RowSource {
  /**
   * コホート iteration の、索引が filter に合う行（貯蔵庫の索引の順）。読み込んでいないコホートなら空。本体は展開しない。
   *
   * @param filter 索引（用途・組・種類・類・測った手の数）の条件
   */
  int[] select(int iteration, Predicate<RowFormat.Index> filter);

  /** 1行を展開する。どのスレッドからでも呼べる。 */
  PrismRow read(int row) throws IOException;
}
