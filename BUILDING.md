# ビルド

JDK 21を用意し、`JAVA_HOME`を設定してください。Mavenは同梱のWrapperを使います。テストも回すビルドでは、PRISMの共通カーネルのテストを組むために、CMake 3.24以降とC++17のコンパイラーも要ります（`-DskipTests`を付けると要りません）。以下のコマンドは、リポジトリのルートで実行します。

```powershell
.\mvnw.cmd package
```

Linux・macOSでは`./mvnw package`を使います。特定のモジュールだけをビルドする場合は、例えばPRISMなら`.\mvnw.cmd -pl epsilon-prism-4p -am package`を実行します。

実行用JARは、それぞれのモジュールの`target/`に生成されます。

| 用途 | 実行用JAR |
| --- | --- |
| Epsilon-Prism | `epsilon-prism-4p/target/epsilon-prism-4p-1.0-SNAPSHOT-all.jar` |
| Epsilon-Prism（三人打ち） | `epsilon-prism-3p/target/epsilon-prism-3p-1.0-SNAPSHOT-all.jar` |
| Epsilon AI レビュアー | `epsilon-reviewer/backend/target/epsilon-reviewer-1.0-SNAPSHOT-all.jar` |

<details>
<summary>AMD GPUを使う場合（Linux / ROCm）</summary>

LinuxでAMD GPUを使う場合は、ROCm環境に合わせてビルドします。ROCm 7.2の例です。

```bash
./mvnw -Procm -Dpytorch.rocm.flavor=rocm7.2 package
```

PRISMのHIPカーネルも同梱する場合は、CMakeとROCmのコンパイラーを用意し、`-Pnative-hip`でビルドします（ROCm用のJNIも選ばれます）。通常用と検査用のライブラリは各アプリの実行用JARに同梱されます。

GPU実行環境の準備は[djl-rocm](https://github.com/KoutaChan/djl-rocm)を参照してください。

</details>

<details>
<summary>NVIDIA GPUを使う場合（Linux / Windows・CUDA）</summary>

NVIDIA GPUを使う場合は、`-Pcuda`でCUDA用のJNIを選びます。PRISMの収集カーネルも同梱する場合は、CMake 3.24以降、CUDA Toolkit 12.8の`nvcc`、対応するC++コンパイラーを用意し、`-Pcuda,native-cuda`でビルドします。四人打ちと三人打ちの例です。

```bash
./mvnw -Pcuda,native-cuda -pl epsilon-prism-4p,epsilon-prism-3p -am package
```

WindowsではVisual Studioのx64開発者コマンドプロンプトで、`./mvnw`を`.\mvnw.cmd`に替えて実行します。コンパイル対象は既定で`sm_120`です。変更する場合は`-Dprism.cuda.arch=<値>`でCMakeのCUDAアーキテクチャ番号を指定します。通常用と検査用のライブラリは各アプリの実行用JARに同梱されます。

CUDA用のJNI（`pytorch-jni-cu128-linux-x86_64`・`pytorch-jni-cu128-win-x86_64`）は、使っているDJLと同じバージョンのものを[djl-rocm](https://github.com/KoutaChan/djl-rocm)で用意してください。

</details>

<details>
<summary>GPUのテスト</summary>

GPUを使うテストは、既定のビルドでは走りません。AMD GPUでは`-Procm-tests`、NVIDIA GPUでは`-Pcuda-tests`を、カーネルのビルドのプロファイルと一緒に付けます。カーネルのnativeのテスト（CTest）は、`-Phip-native-tests`か`-Pcuda-native-tests`で回します。配布モデルのスコアを照らすテストには、配布モデルの置き場を`-Dprism.publishedModels=<ディレクトリ>`で渡します。

```bash
./mvnw -Pnative-hip,rocm-tests,hip-native-tests -Dprism.publishedModels=<ディレクトリ> -pl epsilon-prism-4p,epsilon-prism-3p -am verify
```

</details>

[シリーズ一覧へ戻る](README.md#epsilon-シリーズ)
