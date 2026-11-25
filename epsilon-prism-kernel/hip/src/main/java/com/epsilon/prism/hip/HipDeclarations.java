package com.epsilon.prism.hip;

import com.epsilon.prism.kernel.NativeDeclarations;

/** 読み込んだ共有ライブラリが、Java と同じ宣言の ABI と JNI の口のバージョンで組まれたことを確かめる。 */
public final class HipDeclarations {
  private HipDeclarations() {}

  public static void requireCompatible() {
    try {
      NativeDeclarations.requireLibrary("epsilon-prism-hip", abiVersion(), jniVersion());
    } catch (UnsatisfiedLinkError e) {
      throw new IllegalStateException(
          "Incompatible PRISM HIP library: missing version entry point", e);
    }
  }

  private static native int abiVersion();

  private static native int jniVersion();
}
