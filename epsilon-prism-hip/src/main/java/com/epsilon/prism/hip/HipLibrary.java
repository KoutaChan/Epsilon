package com.epsilon.prism.hip;

import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.jni.LibUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * {@code libprism_hip.so} を読み込み、DJL が読んだ libtorch と同じ HIP のランタイムに繋がったことを確かめる。
 *
 * <p>先に DJL の PyTorch のエンジンを初期化して、libtorch と HIP のランタイムを読ませる。その後に jar の中の .so を
 * {@code ~/.cache/epsilon/prism-hip/<sha>/} に取り出して {@link System#load} する。
 *
 * <p>libprism_hip.so は libtorch のヘッダーを使わず、HIP のランタイム
 * （{@code libamdhip64.so.7}）だけに繋がる。DJL の配列のアドレスだけは、libtorch が公開するシンボルを dlsym で引く。ストリームは
 * {@code PtStream.getId()} の値で、ホストのメモリは direct な ByteBuffer で受け取る。デバイスのメモリは計画が一度だけ確保する。
 * そのため、読み込むときには次のことを照合する。arch は、計画を作るときに GPU ごとに照合する。
 *
 * <ul>
 *   <li>DJL の torch のバージョンが、ビルドしたときのバージョンと同じこと
 *   <li>{@code libtorch_hip.so} が DJL の libtorch の場所から読まれていること
 *   <li>プロセスの HIP のランタイムが 1 つだけで、libprism_hip.so がそれに繋がったこと
 *   <li>HIP のランタイムのバージョン（major.minor）が、ビルドしたときの HIP と libtorch の flavor に合うこと
 *   <li>CXX11 ABI でビルドされていて、検査つきのビルドかどうかが求めたとおりであること
 *   <li>Java と native で、enum の数と、形の記述・WeightPack のヘッダーの大きさが同じこと
 * </ul>
 */
public final class HipLibrary {

  /** 検査つきのビルド（{@code -DPRISM_HIP_CHECK}。グローバルメモリの添字を検査する）を読み込むときに true にするシステムプロパティ。 */
  public static final String CHECKED_PROPERTY = "epsilon.prism.hip.checked";

  private static final String RESOURCES = "/native/linux-x86_64/";
  private static final String RELEASE_LIBRARY = "libprism_hip.so";
  private static final String CHECKED_LIBRARY = "libprism_hip_check.so";
  private static final String BUILD_PROPERTIES = "prism-hip.properties";
  private static final String TORCH_HIP = "libtorch_hip.so";

  private static HipLibrary loaded;

  private final BuildInfo build;
  private final RuntimeInfo runtime;

  private HipLibrary(BuildInfo build, RuntimeInfo runtime) {
    this.build = build;
    this.runtime = runtime;
  }

  /** 読み込む。2 回目からは同じものを返す。検査つきのビルドを読むかは {@link #CHECKED_PROPERTY} で選ぶ。 */
  public static synchronized HipLibrary load() {
    if (loaded == null) {
      loaded = open(Boolean.getBoolean(CHECKED_PROPERTY));
    }
    return loaded;
  }

  /**
   * {@link #load} が読む .so（{@link #CHECKED_PROPERTY} で選んだ方）が、この jar に同梱されているかを返す。-Pnative-hip を付けて
   * ビルドした jar なら true。読み込めるか、DJL の libtorch と HIP のランタイムに合うかは、{@link #load} が確かめる。
   */
  public static boolean bundled() {
    String name = Boolean.getBoolean(CHECKED_PROPERTY) ? CHECKED_LIBRARY : RELEASE_LIBRARY;
    return HipLibrary.class.getResource(RESOURCES + name) != null;
  }

  public RuntimeInfo runtime() {
    return runtime;
  }

  /** 指定した GPU の arch が、ビルドしたときの arch と合うことを確かめる。 */
  void requireArchitecture(int device) {
    String arch = PrismHip.deviceArch(device);
    String expected = build.architecture();
    if (!arch.equals(expected) && !arch.startsWith(expected + ":")) {
      throw new IllegalStateException(
          "GPU " + device + " is " + arch + " but libprism_hip.so was built for " + expected);
    }
  }

  private static HipLibrary open(boolean checked) {
    String os = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");
    if (!"Linux".equals(os) || !("amd64".equals(arch) || "x86_64".equals(arch))) {
      throw new IllegalStateException("libprism_hip.so is built for linux-x86_64 only, not " + os + "/" + arch);
    }
    Engine engine = Engine.getEngine(PtEngine.ENGINE_NAME);
    BuildInfo build = BuildInfo.read();
    Path library = extract(checked ? CHECKED_LIBRARY : RELEASE_LIBRARY);
    System.load(library.toString());
    RuntimeInfo runtime = RuntimeInfo.parse(PrismHip.runtimeInfo());
    verify(build, runtime, engine.getVersion(), realPath(Path.of(LibUtils.getLibtorchPath())), checked);
    return new HipLibrary(build, runtime);
  }

  /**
   * ビルドの条件と、実際に読み込まれたものを照合する。合わなければ {@code IllegalStateException} を投げる。
   *
   * @param build ビルドの条件
   * @param runtime 実際に読み込まれたライブラリの情報
   * @param torchVersion DJL が読んだ PyTorch のバージョン
   * @param libtorchDirectory DJL が読んだ libtorch のディレクトリ（実パス）
   * @param checked 検査つきのビルドを求めたか
   */
  static void verify(
      BuildInfo build, RuntimeInfo runtime, String torchVersion, Path libtorchDirectory, boolean checked) {
    if (!build.torchVersion().equals(torchVersion)) {
      throw new IllegalStateException(
          "DJL loaded PyTorch " + torchVersion + " but libprism_hip.so was built for " + build.torchVersion());
    }
    Path torchHip =
        runtime.torchLoaded().stream()
            .filter(path -> path.getFileName().toString().equals(TORCH_HIP))
            .findFirst()
            .orElseThrow(
                () ->
                    new IllegalStateException(
                        "DJL has not loaded the ROCm libtorch (" + TORCH_HIP + "); loaded " + runtime.torchLoaded()));
    if (!torchHip.getParent().equals(libtorchDirectory)) {
      throw new IllegalStateException(
          TORCH_HIP + " is loaded from " + torchHip.getParent() + ", not from DJL's libtorch " + libtorchDirectory);
    }
    if (runtime.hipLoaded().size() != 1) {
      throw new IllegalStateException("expected exactly one HIP runtime in the process, found " + runtime.hipLoaded());
    }
    if (!runtime.hipLoaded().get(0).equals(runtime.hipLibrary())) {
      throw new IllegalStateException(
          "libprism_hip.so is bound to the HIP runtime "
              + runtime.hipLibrary()
              + " but the process uses "
              + runtime.hipLoaded().get(0));
    }
    if (runtime.hipRuntime() < 0) {
      throw new IllegalStateException("the HIP runtime is unavailable: " + runtime.hipError());
    }
    if (release(runtime.hipRuntime()) != release(runtime.hipCompiled())) {
      throw new IllegalStateException(
          "HIP runtime "
              + releaseName(runtime.hipRuntime())
              + " does not match HIP "
              + releaseName(runtime.hipCompiled())
              + " that libprism_hip.so was built with");
    }
    String flavor = "rocm" + releaseName(runtime.hipRuntime());
    if (!build.torchFlavor().equals(flavor)) {
      throw new IllegalStateException(
          "libprism_hip.so was built for the libtorch flavor " + build.torchFlavor() + " but the HIP runtime is " + flavor);
    }
    if (!runtime.cxx11Abi()) {
      throw new IllegalStateException("libprism_hip.so must use the CXX11 ABI like libtorch " + torchVersion);
    }
    if (runtime.checked() != checked) {
      throw new IllegalStateException(
          "requested the " + (checked ? "checked" : "release") + " build but loaded the other one");
    }
    if (!runtime.architecture().equals(build.architecture())) {
      throw new IllegalStateException(
          "libprism_hip.so reports arch " + runtime.architecture() + " but its build says " + build.architecture());
    }
    if (runtime.kernels() != HipKernel.values().length
        || runtime.regions() != HipRegion.values().length
        || runtime.architectureInts() != HipArchitecture.INTS
        || runtime.headerInts() != HipPack.HEADER_INTS) {
      throw new IllegalStateException(
          "libprism_hip.so has "
              + runtime.kernels()
              + " kernels, "
              + runtime.regions()
              + " regions, a "
              + runtime.architectureInts()
              + "-int architecture and a "
              + runtime.headerInts()
              + "-int weight pack header but Java expects "
              + HipKernel.values().length
              + ", "
              + HipRegion.values().length
              + ", "
              + HipArchitecture.INTS
              + " and "
              + HipPack.HEADER_INTS
              + "; rebuild epsilon-prism-hip");
    }
  }

  /** HIP のバージョンを major·100 + minor にして返す（HIP_VERSION = major·10⁷ + minor·10⁵ + patch）。 */
  private static int release(int hipVersion) {
    return hipVersion / 100_000;
  }

  private static String releaseName(int hipVersion) {
    return hipVersion / 10_000_000 + "." + hipVersion / 100_000 % 100;
  }

  private static Path extract(String name) {
    byte[] bytes = resource(name);
    Path directory = Path.of(System.getProperty("user.home"), ".cache", "epsilon", "prism-hip", digest(bytes));
    Path target = directory.resolve(name);
    try {
      if (!Files.isRegularFile(target) || Files.size(target) != bytes.length) {
        Files.createDirectories(directory);
        Path partial = Files.createTempFile(directory, name, ".part");
        Files.write(partial, bytes);
        Files.move(partial, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
      }
    } catch (IOException e) {
      throw new UncheckedIOException("cannot extract " + name + " to " + directory, e);
    }
    return target;
  }

  private static byte[] resource(String name) {
    try (InputStream in = HipLibrary.class.getResourceAsStream(RESOURCES + name)) {
      if (in == null) {
        throw new IllegalStateException(
            RESOURCES + name + " is not on the classpath; build epsilon-prism-hip with -Pnative-hip on the server");
      }
      return in.readAllBytes();
    } catch (IOException e) {
      throw new UncheckedIOException("cannot read " + RESOURCES + name, e);
    }
  }

  private static String digest(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)).substring(0, 16);
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 is unavailable", e);
    }
  }

  private static Path realPath(Path path) {
    try {
      return path.toRealPath();
    } catch (IOException e) {
      throw new UncheckedIOException("cannot resolve " + path, e);
    }
  }

  /**
   * libprism_hip.so をビルドした条件（{@code prism-hip.properties} から読む）。
   *
   * @param torchVersion ビルドで合わせた PyTorch のバージョン（DJL のバージョンと同じであること）
   * @param torchFlavor ビルドで合わせた libtorch の flavor（{@code rocm<major>.<minor>}）
   * @param architecture ビルドの対象の GPU の arch
   */
  public record BuildInfo(String torchVersion, String torchFlavor, String architecture) {

    static BuildInfo read() {
      Properties properties = new Properties();
      try (InputStream in = HipLibrary.class.getResourceAsStream(RESOURCES + BUILD_PROPERTIES)) {
        if (in == null) {
          throw new IllegalStateException(RESOURCES + BUILD_PROPERTIES + " is not on the classpath");
        }
        properties.load(in);
      } catch (IOException e) {
        throw new UncheckedIOException("cannot read " + BUILD_PROPERTIES, e);
      }
      return new BuildInfo(
          required(properties, "torch.version"), required(properties, "torch.flavor"), required(properties, "arch"));
    }

    private static String required(Properties properties, String key) {
      String value = properties.getProperty(key);
      if (value == null || value.isBlank()) {
        throw new IllegalStateException(BUILD_PROPERTIES + " has no " + key);
      }
      return value.trim();
    }
  }

  /**
   * 実際に読み込まれたライブラリの情報（native の {@code runtimeInfo()} が返す）。パスは全て実パス。
   *
   * @param hipCompiled ビルドしたときの HIP_VERSION
   * @param hipRuntime ランタイムのバージョン（取れなければ −1）
   * @param hipError ランタイムのバージョンが取れなかった理由（取れたら空）
   * @param hipLibrary libprism_hip.so が繋がった libamdhip64
   * @param hipLoaded プロセスにある全ての libamdhip64
   * @param torchLoaded プロセスにある libtorch*・libc10*
   * @param cxx11Abi CXX11 ABI でビルドしたか
   * @param checked 検査つきのビルドか
   * @param kernels native のカーネルの数
   * @param regions native が作業領域の中に持つ領域の数
   * @param architectureInts native の形の記述の int の数
   * @param headerInts native の WeightPack のヘッダーの int の数
   * @param architecture ビルドの対象の arch
   */
  public record RuntimeInfo(
      int hipCompiled,
      int hipRuntime,
      String hipError,
      Path hipLibrary,
      List<Path> hipLoaded,
      List<Path> torchLoaded,
      boolean cxx11Abi,
      boolean checked,
      int kernels,
      int regions,
      int architectureInts,
      int headerInts,
      String architecture) {

    public RuntimeInfo {
      hipLoaded = List.copyOf(hipLoaded);
      torchLoaded = List.copyOf(torchLoaded);
    }

    static RuntimeInfo parse(String[] pairs) {
      Map<String, String> values = new HashMap<>();
      for (String pair : pairs) {
        int equals = pair.indexOf('=');
        if (equals <= 0) {
          throw new IllegalStateException("malformed runtime fact from libprism_hip.so: " + pair);
        }
        values.put(pair.substring(0, equals), pair.substring(equals + 1));
      }
      return new RuntimeInfo(
          Integer.parseInt(required(values, "hip.compiled")),
          Integer.parseInt(required(values, "hip.runtime")),
          values.getOrDefault("hip.error", ""),
          Path.of(required(values, "hip.library")),
          paths(required(values, "hip.loaded")),
          paths(required(values, "torch.loaded")),
          "1".equals(required(values, "cxx11abi")),
          Boolean.parseBoolean(required(values, "checked")),
          Integer.parseInt(required(values, "kernels")),
          Integer.parseInt(required(values, "regions")),
          Integer.parseInt(required(values, "architecture")),
          Integer.parseInt(required(values, "header")),
          required(values, "arch"));
    }

    private static String required(Map<String, String> values, String key) {
      String value = values.get(key);
      if (value == null) {
        throw new IllegalStateException("libprism_hip.so did not report " + key);
      }
      return value;
    }

    private static List<Path> paths(String joined) {
      return joined.isEmpty() ? List.of() : Arrays.stream(joined.split(":")).map(Path::of).toList();
    }
  }
}
