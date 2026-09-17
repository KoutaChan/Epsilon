package com.epsilon.ai.grp;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.training.optimizer.Optimizer;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * GRP の AdamW 状態を、プロセスごとに変わる DJL UID ではなく階層パラメーター名に対応させる。
 *
 * <p>旧 GRP v3 は全パラメーターを毎回更新する。旧 UID は生成時の単調カウンター順に、現在の同じ GRP 構成の UID 生成順へ対応させる。GRU の生成順と
 * getParameters() の列挙順は同一視しない。 全パラメーターのカウンターと一次・二次モーメントが揃い、形状とデータ型が一致する状態だけを復元する。 一部だけの状態には欠けた UID
 * の位置を確定する情報がないため、推測せず拒否する。
 */
final class GrpOptimizer extends Optimizer {
  private static final String MAGIC = "DJL_OPTIMIZER_STATE";
  private static final String ADAMW = "ai.djl.training.optimizer.AdamW";

  private final Optimizer delegate;
  private final Map<String, String> namesById = new HashMap<>();
  private final Map<String, Parameter> parameters = new LinkedHashMap<>();

  GrpOptimizer(ParameterList modelParameters, Optimizer delegate) {
    super(Optimizer.adamW());
    this.delegate = delegate;
    for (var parameter : modelParameters) {
      namesById.put(parameter.getValue().getId(), parameter.getKey());
      parameters.put(parameter.getKey(), parameter.getValue());
    }
  }

  @Override
  public void update(String parameterId, NDArray weight, NDArray grad) {
    String name = namesById.get(parameterId);
    if (name == null) {
      throw new IllegalArgumentException("Unknown GRP parameter: " + parameterId);
    }
    delegate.update(name, weight, grad);
  }

  @Override
  public void saveState(Path path) throws IOException {
    delegate.saveState(path);
  }

  @Override
  public void loadState(NDManager manager, Path path) throws IOException {
    try (DataInputStream input = new DataInputStream(Files.newInputStream(path));
        NDManager decoding = manager.newSubManager(Device.cpu())) {
      if (!MAGIC.equals(input.readUTF())
          || input.readInt() != 1
          || !ADAMW.equals(input.readUTF())) {
        throw new IOException("Unsupported GRP AdamW state: " + path);
      }
      int beginUpdate = input.readInt();
      int update = input.readInt();
      int size = input.readInt();
      if (beginUpdate != 0
          || update < 0
          || (size != parameters.size() && !(size == 0 && update == 0))) {
        throw new IOException("Incomplete GRP optimizer counters: " + path);
      }
      Map<String, Integer> counts = new LinkedHashMap<>();
      for (int i = 0; i < size; i++) {
        String key = input.readUTF();
        int count = input.readInt();
        if (count != update || count <= 0 || counts.put(key, count) != null) {
          throw new IOException("Inconsistent GRP optimizer counter: " + key);
        }
      }
      Map<String, String> restoredNames = restoredNames(counts.keySet());
      int byteLength = input.readInt();
      if (byteLength < 0 || byteLength > Files.size(path)) {
        throw new IOException("Invalid GRP optimizer array length: " + byteLength);
      }
      byte[] bytes = input.readNBytes(byteLength);
      if (bytes.length != byteLength || input.read() != -1) {
        throw new IOException("Invalid GRP optimizer state length: " + path);
      }
      try (NDList arrays = NDList.decode(decoding, new ByteArrayInputStream(bytes))) {
        Set<String> moments = new HashSet<>();
        String sourceDevice = null;
        for (NDArray array : arrays) {
          String[] key = array.getName().split("\\.", -1);
          if (key.length != 4) {
            throw new IOException("Invalid GRP optimizer array key: " + array.getName());
          }
          String state = decode(key[0]);
          String name = restoredNames.get(decode(key[1]));
          String device = decode(key[2]) + ':' + Integer.parseInt(key[3]);
          if (sourceDevice == null) sourceDevice = device;
          Parameter parameter = parameters.get(name);
          if ((!state.equals("means") && !state.equals("variances"))
              || parameter == null
              || !moments.add(state + ':' + name)
              || !sourceDevice.equals(device)
              || !parameter.getArray().getShape().equals(array.getShape())
              || parameter.getArray().getDataType() != array.getDataType()) {
            throw new IOException("Incompatible GRP optimizer moment: " + array.getName());
          }
          Device target = parameter.getArray().getDevice();
          array.setName(
              encode(state)
                  + '.'
                  + encode(name)
                  + '.'
                  + encode(target.getDeviceType())
                  + '.'
                  + target.getDeviceId());
        }
        if (moments.size() != size * 2) {
          throw new IOException("Incomplete GRP optimizer moments: " + path);
        }
        // DJL の既存保存形式と AdamW 実装を維持し、復元時のキーだけを変換する。
        Path normalized = Files.createTempFile("epsilon-grp-optimizer-", ".state");
        try {
          try (DataOutputStream output = new DataOutputStream(Files.newOutputStream(normalized));
              ByteArrayOutputStream encoded = new ByteArrayOutputStream()) {
            output.writeUTF(MAGIC);
            output.writeInt(1);
            output.writeUTF(ADAMW);
            output.writeInt(beginUpdate);
            output.writeInt(update);
            output.writeInt(size);
            for (var count : counts.entrySet()) {
              output.writeUTF(restoredNames.get(count.getKey()));
              output.writeInt(count.getValue());
            }
            arrays.encode(encoded, NDList.Encoding.NPZ);
            output.writeInt(encoded.size());
            encoded.writeTo(output);
          }
          delegate.loadState(manager, normalized);
        } finally {
          Files.deleteIfExists(normalized);
        }
      }
    } catch (IllegalArgumentException e) {
      throw new IOException("Invalid GRP optimizer state: " + path, e);
    }
  }

  private Map<String, String> restoredNames(Set<String> savedKeys) throws IOException {
    Map<String, String> names = new HashMap<>();
    if (savedKeys.isEmpty()) return names;
    if (parameters.keySet().equals(savedKeys)) {
      for (String key : savedKeys) names.put(key, key);
      return names;
    }
    ArrayList<String> savedIds = new ArrayList<>(savedKeys);
    ArrayList<String> currentIds = new ArrayList<>(namesById.keySet());
    Set<Long> savedCounters = new HashSet<>();
    for (String id : savedIds) {
      if (!isUid(id) || !savedCounters.add(uidCounter(id))) {
        throw new IOException("Unknown or ambiguous GRP optimizer parameter ID: " + id);
      }
    }
    for (String id : currentIds) {
      if (!isUid(id)) throw new IOException("Unsupported DJL parameter ID: " + id);
    }
    savedIds.sort(Comparator.comparingLong(GrpOptimizer::uidCounter));
    currentIds.sort(Comparator.comparingLong(GrpOptimizer::uidCounter));
    for (int i = 0; i < savedIds.size(); i++) {
      names.put(savedIds.get(i), namesById.get(currentIds.get(i)));
    }
    return names;
  }

  private static boolean isUid(String id) {
    return id.matches("uid--?\\d+-\\d+");
  }

  private static long uidCounter(String id) {
    return Long.parseLong(id.substring(id.lastIndexOf('-') + 1));
  }

  private static String encode(String value) {
    return Base64.getUrlEncoder()
        .withoutPadding()
        .encodeToString(value.getBytes(StandardCharsets.UTF_8));
  }

  private static String decode(String value) {
    return new String(Base64.getUrlDecoder().decode(value), StandardCharsets.UTF_8);
  }
}
