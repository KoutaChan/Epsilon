package com.epsilon.agent.runtime;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;

/** 実行のセクション。JVM を起動したとき、PyTorch を読み込む前に読む。 */
public record RuntimeSettings(
    @Key(doc = "PyTorch が CPU の演算に使うスレッドの数", value = "1", change = Change.MACHINE) @Range(min = 1)
        int torchThreads) {

  /** PyTorch が CPU の演算に使うスレッド数を、DJL が読むシステムプロパティ。 */
  static final String TORCH_THREADS = "ai.djl.pytorch.num_threads";

  /** 演算のスレッド数を DJL に渡す。この JVM で PyTorch を使う前に呼ぶ。PyTorch を読み込んだ後では効かない。 */
  public void apply() {
    System.setProperty(TORCH_THREADS, Integer.toString(torchThreads));
  }
}
