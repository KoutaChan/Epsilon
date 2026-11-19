package com.epsilon.prism.rows;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import org.testng.Assert;

/** コホートの行と対応するインデックスを、書き込み順に依存せず比較する。 */
public record CohortRows(String body, String index) {
  public static List<CohortRows> read(Path cohort) throws IOException {
    var manifest = CohortWriter.readManifest(cohort);
    var codec = new RowCodec(manifest.ledgerFeatures(), manifest.variant());
    var rows = new ArrayList<CohortRows>();
    var hex = HexFormat.of();
    for (var shard : manifest.shards()) {
      Path file = cohort.resolve(shard.file());
      var body = ByteBuffer.wrap(Files.readAllBytes(file));
      var index =
          ByteBuffer.wrap(Files.readAllBytes(CohortWriter.index(file)))
              .order(ByteOrder.LITTLE_ENDIAN);
      while (body.hasRemaining()) {
        int start = body.position();
        codec.decode(body);
        int entry = index.position();
        Assert.assertEquals(index.getInt(entry), body.position() - start, file.toString());
        index.position(entry + CohortWriter.INDEX_ENTRY);
        rows.add(
            new CohortRows(
                hex.formatHex(body.array(), start, body.position()),
                hex.formatHex(index.array(), entry, index.position())));
      }
      Assert.assertFalse(index.hasRemaining(), file.toString());
    }
    Assert.assertEquals((long) rows.size(), manifest.rows());
    rows.sort(Comparator.comparing(CohortRows::body).thenComparing(CohortRows::index));
    return rows;
  }
}
