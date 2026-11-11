<div align="center">

# Epsilon Reviewer

牌譜をAIで解析し、判断ごとの候補手を確認できるレビュー画面。

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![Electron](https://img.shields.io/badge/Electron-47848F?logo=electron&logoColor=white)](https://www.electronjs.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

</div>

PC・スマホのブラウザーと、Windowsアプリで使えます。天鳳・mjai・雀魂の四人麻雀と、天鳳の三人麻雀の牌譜を、Epsilon-Prism・Epsilon-Nano・Epsilon-Majorのモデルで解析します。候補手の上位3件と最善手との損失を示し、「もしも」で局面を変えて解析し直せます。Web版の解析結果には共有リンクが付きます。

## はじめに

Javaのビルドは[プロジェクト共通の手順](../BUILDING.md)を参照してください。画面にはNode.js 24を使います。リポジトリのルートでエンジンと画面をビルドし、[設定例](backend/models.example.json)を写してモデルの`checkpoint`を書き換え、サーバーを起動します。

```powershell
.\mvnw.cmd -pl epsilon-reviewer/backend -am -DskipTests package
cd epsilon-reviewer
npm ci
npm run build
Copy-Item backend/models.example.json models.local.json
npm run server
```

[http://127.0.0.1:8080](http://127.0.0.1:8080)を開き、牌譜とモデルを選んで「解析する」を押します。サーバーを終了するには`Ctrl+C`を押します。

<div align="center">

[シリーズ一覧](../README.md) / [ライセンス](../LICENSE) / [第三者ライセンス](../LICENSES/README.md)

</div>
