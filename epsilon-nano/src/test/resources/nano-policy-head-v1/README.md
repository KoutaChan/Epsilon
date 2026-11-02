Nanoの候補表現抽出前の`EpsilonDecisionPolicyHead`から作った互換性fixture。

元のソースは抽出作業開始時のコミット `46f2813457735e9cc3824791fb642e8bd3cffafd` のPolicyHeadを別class directoryへコンパイルし、通常のNano runtimeより先にclasspathへ置いて実行した。`NetworkFactory.createDecisionModel(cpu, false, 8)`、PyTorch seed `31007`で初期化して`nano-0000.params`を保存した。

`expected.json`は全parameter名と、GameEngine seeds `772`/`817`の各16 decision boundaryについて、毎回最初の合法手を選択したときの3種類の方策スコア。境界の入力は既存`DecisionBatchBuilder.addInferenceRow`、境界価値は`DecisionBoundaryContext.uniform()`。新実装から期待値を再生成しない。
