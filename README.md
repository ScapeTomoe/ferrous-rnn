# ferrous-rnn

Rustで実装されたRNNライブラリです。PyO3を通じてPythonから利用できます。

## 特徴

- Rustによる高速・安全な実装
- 一部の演算をインラインアセンブリで最適化
- Pythonから簡単に呼び出し可能

## セットアップ

```bash
pip install maturin
maturin develop
```

## 使い方

```python
import rnn_core

# モデルの作成
rnn = rnn_core.PyRnn(
    input_dim=10,      # 入力の次元数
    hidden_dim=32,     # 隠れ状態の次元数
    activation="tanh"  # 活性化関数
)

# 全タイムステップを返す
result = rnn.forward(x, sequences=True)

# 最終タイムステップのみ返す
result = rnn.forward(x, sequences=False)
```

## 使える活性化関数

| 名前 | 文字列指定 |
|---|---|
| Tanh | `"tanh"` |
| ReLU | `"relu"` |
| Sigmoid | `"sigmoid"` |
| Leaky ReLU | `"leaky_relu"` |
| ELU | `"elu"` |

## ファイル構成

```
src/
├── lib.rs          # モジュールの公開
├── activation.rs   # 活性化関数
├── rnn.rs          # RNN本体
├── params.rs       # 学習用勾配
└── python/
    └── rnn_py.rs   # Pythonバインディング
```

## 注意事項

- x86_64アーキテクチャ（SSE2）を前提としています
- 入力行列は `(seq_len, input_dim)` の形式で渡してください
- 不正な活性化関数名を渡すと `ValueError` が発生します
