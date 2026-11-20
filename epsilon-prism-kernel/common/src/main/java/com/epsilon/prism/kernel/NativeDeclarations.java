package com.epsilon.prism.kernel;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/** 読み込んだ共有ライブラリのモデル宣言の ABI と JNI の口のバージョンを照合する。 */
public final class NativeDeclarations {
  private NativeDeclarations() {}

  static String resource(String name) {
    try (var in = NativeDeclarations.class.getResourceAsStream(name)) {
      if (in == null) throw new IllegalStateException("Missing PRISM version resource: " + name);
      var value = new String(in.readAllBytes(), StandardCharsets.UTF_8).strip();
      if (value.isEmpty()) throw new IllegalStateException("Empty PRISM version resource: " + name);
      return value;
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  /** JNI の口（native のメソッドの名前と引数）のバージョン。共有ライブラリと同じ値でなければならない。口を変えたら上げる。 */
  public static final int JNI_VERSION = Integer.parseInt(resource("jni_version.inc"));

  /**
   * 読み込んだ共有ライブラリのモデル宣言の ABI と JNI の口のバージョンを、Java の側と照合する。
   *
   * @param artifact 共有ライブラリを配る Maven のモジュール（例外の文に使う）
   * @param abi 共有ライブラリが返すモデル宣言の ABI
   * @param jni 共有ライブラリが返す JNI の口のバージョン
   * @throws IllegalStateException どちらかが Java の側と違うとき
   */
  public static void requireLibrary(String artifact, int abi, int jni) {
    if (abi != ModelSchema.VERSION)
      throw new IllegalStateException(
          "PRISM declaration ABI mismatch for "
              + artifact
              + ": expected "
              + ModelSchema.VERSION
              + ", got "
              + abi);
    if (jni != JNI_VERSION)
      throw new IllegalStateException(
          "PRISM JNI version mismatch for "
              + artifact
              + ": expected "
              + JNI_VERSION
              + ", got "
              + jni);
  }
}
