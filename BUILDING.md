# ビルド

JDK 21を用意し、`JAVA_HOME`を設定してください。Mavenは同梱のWrapperを使います。以下のコマンドは、リポジトリのルートで実行します。

```powershell
.\mvnw.cmd package
```

Linux・macOSでは`./mvnw package`を使います。特定の系列だけをビルドする場合は、例えばNanoなら`.\mvnw.cmd -pl epsilon-nano -am package`を実行します。

実行用JARは、それぞれのモジュールの`target/`に生成されます。

| 系列           | 実行用JAR                                                  |
|--------------|---------------------------------------------------------|
| Epsilon-Pico | `epsilon-pico/target/epsilon-pico-1.0-SNAPSHOT-all.jar` |
| Epsilon-Nano | `epsilon-nano/target/epsilon-nano-1.0-SNAPSHOT-all.jar` |
| Epsilon      | `epsilon/target/epsilon-1.0-SNAPSHOT-all.jar`           |
| Epsilon-Prism | `epsilon-prism/target/epsilon-prism-1.0-SNAPSHOT-all.jar` |

<details>
<summary>AMD GPUを使う場合（Linux / ROCm）</summary>

LinuxでAMD GPUを使う場合は、ROCm環境に合わせてビルドします。ROCm 7.2の例です。

```bash
./mvnw -Procm -Dpytorch.rocm.flavor=rocm7.2 package
```

GPU実行環境の準備は[djl-rocm](https://github.com/KoutaChan/djl-rocm)を参照してください。

</details>

<details>
<summary>NVIDIA GPUを使う場合（Linux / Windows・CUDA）</summary>

NVIDIA GPUを使う場合は、`-Pcuda`を付けてビルドします。実行用JARに同梱するDJLのJNIがCUDA版になります。CUDA 12.8の例です。

```bash
./mvnw -Pcuda -Dpytorch.cuda.flavor=cu128 package
```

CUDA版のJNI（`pytorch-jni-cu128-linux-x86_64`・`pytorch-jni-cu128-win-x86_64`）は、使っているDJLと同じ版のものを[djl-rocm](https://github.com/KoutaChan/djl-rocm)で用意する必要があります。

</details>

[シリーズ一覧へ戻る](README.md#epsilon-シリーズ)
