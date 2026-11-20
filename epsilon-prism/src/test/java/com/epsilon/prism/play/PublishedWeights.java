package com.epsilon.prism.play;

import static org.testng.Assert.assertEquals;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.kernel.ModelDeclarations;
import com.epsilon.prism.kernel.ModelSchema;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.util.Properties;
import java.util.zip.CRC32;

/**
 * 配布モデルを CPU で読み、宣言の往復、パラメーターの名前・形・順、専用カーネルに渡す pack の長さと CRC-32 を確かめる。pack の長さと CRC-32 は、
 * 配布したときの pack の固定値（{@value #PACKS}）と同じでなければならない。
 */
public final class PublishedWeights {
  private static final String PACKS = "published-packs.properties";

  private PublishedWeights() {}

  public static void check(PublishedModel model) throws IOException {
    try (var manager = NDManager.newBaseManager(Device.cpu())) {
      var network = PrismNetwork.load(Snapshot.model(model.snapshot()), manager);
      var declaration = network.declaration();
      var encoded = ModelDeclarations.encode(declaration);
      assertEquals(
          ModelDeclarations.decode(encoded, encoded.remaining(), ModelSchema.VERSION), declaration);
      assertEquals(declaration.seats() + declaration.tiles() + 1, network.format().fixedTokens());
      var params = network.inferenceParameters();
      assertEquals(params.size(), declaration.weights().size());
      for (int i = 0; i < params.size(); i++) {
        var weight = declaration.weights().get(i);
        assertEquals(weight.name(), params.get(i).getKey());
        assertEquals(
            params.get(i).getValue().getArray().getShape().getShape(),
            weight.axes().stream().mapToLong(Integer::longValue).toArray(),
            weight.name());
      }
      var data = WeightPack.publication(network, 7).data();
      byte[] bytes = new byte[data.remaining()];
      data.get(bytes);
      var crc = new CRC32();
      crc.update(bytes);
      var packs = packs();
      assertEquals(bytes.length, Long.parseLong(packs.getProperty(model.directory() + ".bytes")));
      assertEquals(crc.getValue(), Long.parseLong(packs.getProperty(model.directory() + ".crc32")));
    }
  }

  private static Properties packs() throws IOException {
    try (var input = PublishedWeights.class.getResourceAsStream(PACKS)) {
      if (input == null) throw new IOException("Missing published pack digests: " + PACKS);
      var packs = new Properties();
      packs.load(input);
      return packs;
    }
  }
}
