package com.epsilon.prism.hip;

/** ランタイムをロードせずに、選択した HIP ライブラリが配布物に含まれるかを調べる。 */
public final class HipResources {
  public static final String CHECKED_PROPERTY = "epsilon.prism.hip.checked";
  static final String DIRECTORY = "/native/linux-x86_64/";
  static final String RELEASE = "libprism_hip.so";
  static final String CHECKED = "libprism_hip_check.so";

  private HipResources() {}

  public static boolean bundled() {
    return HipResources.class.getResource(DIRECTORY + selected()) != null;
  }

  /** {@link #CHECKED_PROPERTY} で選んだ共有ライブラリのファイル名。 */
  static String selected() {
    return Boolean.getBoolean(CHECKED_PROPERTY) ? CHECKED : RELEASE;
  }
}
