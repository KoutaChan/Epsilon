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

<details>
<summary>AMD GPUを使う場合（Linux / ROCm）</summary>

LinuxでAMD GPUを使う場合は、ROCm環境に合わせてビルドします。ROCm 7.2の例です。

```bash
./mvnw -Procm -Dpytorch.rocm.flavor=rocm7.2 package
```

GPU実行環境の準備は[djl-rocm](https://github.com/KoutaChan/djl-rocm)を参照してください。

</details>

[シリーズ一覧へ戻る](README.md#epsilon-シリーズ)
