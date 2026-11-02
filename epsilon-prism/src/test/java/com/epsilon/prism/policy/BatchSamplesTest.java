package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.run.Snapshot;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/**
 * 受け付けの抜き取り: K 個目ごとに N 個までだけ書き、読み戻した O が問い合わせの O と同じで、そこから作り直した塊の blob が問い合わせの順の
 * 塊（方策が HIP で 1 塊にするときと同じ）の blob とバイトで同じ。設定がなければ何もせず、形が違えば例外。
 */
public class BatchSamplesTest {

  private static final int GAMES = 6;
  private static final int BATCHES = 5;
  private static final int VERSION = 7;

  @Test
  public void everyKthAdmissionIsWrittenAndReplaysTheSameBlob() throws Exception {
    Path root = Files.createTempDirectory("prism-batch-samples");
    try {
      BatchSamples samples = BatchSamples.of("2:1:" + root);
      List<List<long[]>> offered = new ArrayList<>();
      byte[][] written = new byte[1][];
      Observation read = new Observation();
      NetworkPolicyTest.walk(
          GAMES,
          BATCHES,
          requests -> {
            List<long[]> words = new ArrayList<>();
            for (PolicyRequest request : requests) {
              int base = request.recordBase();
              int longs = read.bind(request.recordWords(), base).longs();
              words.add(Arrays.copyOfRange(request.recordWords(), base, base + longs));
            }
            offered.add(words);
            if (offered.size() == 2) {
              // 区画の間の詰め物も比べるので、使い回していない塊に書く
              written[0] = bytes(new RequestBatch(false).encode(requests).input());
            }
            samples.offer(requests, VERSION);
          });
      assertTrue(offered.size() >= 4, "the walk must offer several admissions: " + offered.size());
      List<Path> files;
      try (Stream<Path> listed = Files.list(root)) {
        files = listed.sorted().toList();
      }
      assertEquals(files.size(), 1, files.toString());
      assertEquals(files.getFirst().getFileName().toString(), "prism-v7-000000002.batch");
      List<long[]> rows = BatchSamples.read(files.getFirst());
      List<long[]> expected = offered.get(1);
      assertEquals(rows.size(), expected.size());
      for (int row = 0; row < rows.size(); row++) {
        assertTrue(Arrays.equals(rows.get(row), expected.get(row)), "row " + row);
      }
      assertTrue(Arrays.equals(bytes(HipFixtures.encode(rows)), written[0]), "replayed blob");
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void settingsAreChecked() {
    BatchSamples.of(null).offer(List.of(), VERSION);
    BatchSamples.of(" ").offer(List.of(), VERSION);
    expectThrows(IllegalArgumentException.class, () -> BatchSamples.of("0:1:/tmp"));
    expectThrows(IllegalArgumentException.class, () -> BatchSamples.of("4:0:/tmp"));
    expectThrows(IllegalArgumentException.class, () -> BatchSamples.of("/tmp"));
  }

  /** 塊の使うバイト（blob の先頭から {@link HostBatch#bytes}）。 */
  private static byte[] bytes(HostBatch batch) {
    byte[] bytes = new byte[batch.bytes()];
    ByteBuffer blob = batch.blob().duplicate();
    blob.position(0);
    blob.get(bytes);
    return bytes;
  }
}
