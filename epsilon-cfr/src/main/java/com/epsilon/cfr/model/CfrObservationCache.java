package com.epsilon.cfr.model;

import com.epsilon.cfr.data.CfrTrainingRow;
import java.util.LinkedHashMap;

/** 学習中に再抽出された同じ行の観測を再利用する。保持する元行と復号結果の合計容量には上限を設ける。 */
final class CfrObservationCache implements AutoCloseable {
  private static final long DEFAULT_MAX_BYTES = 64L << 20;
  private static final int DEFAULT_MAX_ROWS = 4096;
  private final long maxBytes;
  private final int maxRows;
  private final LinkedHashMap<Identity, Entry> entries = new LinkedHashMap<>(16, .75f, true);
  private long retainedBytes;

  CfrObservationCache() {
    this(DEFAULT_MAX_BYTES, DEFAULT_MAX_ROWS);
  }

  CfrObservationCache(long maxBytes, int maxRows) {
    if (maxBytes < 0 || maxRows < 0)
      throw new IllegalArgumentException("Invalid CFR observation cache capacity");
    this.maxBytes = maxBytes;
    this.maxRows = maxRows;
  }

  /** 返却した配列とホストバッチは読み取り専用として扱う。同じ標本IDでも、別の行オブジェクトなら必ず復号する。 */
  CfrObservationCodec.Decoded get(CfrTrainingRow row) {
    var identity = new Identity(row);
    Entry cached = entries.get(identity);
    if (cached != null) return cached.observation();
    var decoded = CfrObservationCodec.decode(row.observation());
    long bytes = retainedBytes(row, decoded);
    if (maxRows == 0 || bytes > maxBytes) return decoded;
    while (entries.size() >= maxRows || retainedBytes > maxBytes - bytes)
      retainedBytes -= entries.pollFirstEntry().getValue().bytes();
    entries.put(identity, new Entry(decoded, bytes));
    retainedBytes += bytes;
    return decoded;
  }

  private static long retainedBytes(CfrTrainingRow row, CfrObservationCodec.Decoded decoded) {
    var inputs = decoded.host().inputs();
    // 元行の観測・教師配列も強参照で残る。配列本体に加え、行・配置・マップなどの管理領域を余裕を持って見積もる。
    return row.serializedBytes()
        + (long) inputs.denseCategories().length * Short.BYTES
        + (long) inputs.denseNumerics().length * Float.BYTES
        + 2048;
  }

  @Override
  public void close() {
    entries.clear();
    retainedBytes = 0;
  }

  private record Entry(CfrObservationCodec.Decoded observation, long bytes) {}

  private record Identity(CfrTrainingRow row) {
    @Override
    public int hashCode() {
      return System.identityHashCode(row);
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Identity identity && row == identity.row;
    }
  }
}
