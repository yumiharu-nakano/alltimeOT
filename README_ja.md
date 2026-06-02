# All-time 最適輸送 — 日本語解説

> 論文 *Continuum-marginal optimal transport: a mesh-free kernel method*（Y. Nakano, 2026）の参照実装。
> [arXiv:2604.24226](https://arxiv.org/abs/2604.24226)

このリポジトリは論文 *All-time Optimal Transport* で提案している
**全時間 OT ドリフト推定器** の参照実装です。時間添字付きの確率分布族
`{μ_t}_{t∈[0,T]}` の「各時刻のサンプル」だけを入力として、
連続の式
```
∂_t μ_t + ∇·(u μ_t) = 0
```
を満たす速度場 `u*(t, x)` を再現することを目的とします。

従来の 2 マージナル OT は `μ_0` と `μ_T` の 2 点しか見ないため、
`μ_0 = μ_T`（周回運動）の場合に `u ≡ 0` と誤った解を返します。
Waddington-OT は多数の中間スナップショット（`M` 個）を使いますが、
ドリフト再構成誤差が `O(M/√N)` で発散するという弱点があります。
**本推定器は中間時刻のサンプルを連続的に活用し、少ないサンプル数でも
非自明なドリフトを安定的に回復する** ことが特長です。

---


## ディレクトリ構成

```
alltime_ot/              共通パッケージ
    rkhs.py              RKHS 全時間 OT 損失 (PyTorch)
    features.py          特徴辞書 (affine / bilinear / tanh basis)
    ensemble.py          scipy L-BFGS-B 用アンサンブル目的関数
    problems.py          ベンチマーク問題のサンプリング関数
    simulate.py          Euler ODE 積分, W_2 / sliced W_2 / MMD 評価
    baselines.py         比較手法 (Affine MMOT, Sinkhorn WOT)

experiments/             各実験スクリプト
    exp1_gaussian.py         Exp 1: 1 次元 Gauss 並進
    exp2_roundtrip.py        Exp 2: 1 次元 周回運動 (線形モデル)
    exp2_mlp.py              Exp 2: 1 次元 周回運動 (MLP ニューラルネット)
    exp2_flow_matching.py    Exp 2: Flow Matching ベースライン
    exp2_mmot_learned_affine.py  Exp 2: アフィン MMOT
    exp2_wot_comparison.py   Exp 2: Waddington-OT 比較
    exp3_bimodal.py          Exp 3: 1 次元 双峰融合フロー（bilinear / tanh辞書 / MLP 比較）
    exp3_baselines.py        Exp 3: 双峰問題での DLA/WOT 比較
    exp4_2d_translation.py   Exp 4: 2 次元 Gauss 並進
    exp5_2d_bifurcation.py   Exp 5: 2 次元 分岐問題
    exp6_eb_preprocess.py            §4.6: 単細胞 RNA-seq (EB) データ前処理
    exp6_eb_alltime_interp.py        §4.6: All-time MLP ドリフト学習（補間設定）
    exp6_eb_baselines_interp.py      §4.6: Waddington-OT・zero drift ベースライン
    exp6_eb_evaluate_interp.py       §4.6: held-out day 15 で SW2 と MMD 評価
    exp6_eb_bootstrap_stability.py   §4.6: 推定速度場のブートストラップ安定性診断
    exp_stochastic_1d_train.py    §4.7: 1 次元 Nelson 問題（アフィンドリフト学習）
    exp_stochastic_1d_metrics.py  §4.7: 1 次元 Nelson 問題のメトリクス（weights.json 読込）
    appA_sensitivity.py      付録 A: (M, N, λ) 感度解析
    appB_dim_scaling.py      付録 B: 次元スケーリング (d=1…10)

requirements.txt
pyproject.toml
.gitignore
LICENSE
README.md / README_ja.md
```

論文本体（LaTeX ソース・PDF）は arXiv で公開しており、本リポジトリには
含めていません（arXiv が single source of truth）。


## 各実験の概要

| ファイル | 問題 | 学習対象 |
|----------|------|----------|
| `exp1_gaussian.py` | `μ_t = N(-1+2t, 1)` | 定ドリフト `u*=2` |
| `exp2_roundtrip.py` | `μ_t = N(2 sin(πt), 1)`, `μ_0=μ_1` | `u*=2π cos(πt)`（線形モデル） |
| `exp2_mlp.py` | 同上 | `u*=2π cos(πt)`（MLP ニューラルネット） |
| `exp2_flow_matching.py` | 同上 | Flow Matching ベースライン |
| `exp2_mmot_learned_affine.py` | 同上 | アフィン DLA ベースライン |
| `exp2_wot_comparison.py` | 同上 | Waddington-OT ベースライン |
| `exp3_bimodal.py` | 双峰 → 合流 | `u*=-2 tanh(2(1-t)x)`（bilinear / tanh 辞書 / MLP を同一損失で比較） |
| `exp3_baselines.py` | 同上 | DLA / WOT ベースライン |
| `exp4_2d_translation.py` | 2 次元 Gauss 並進 | `u*=(2, 0.5)` |
| `exp5_2d_bifurcation.py` | 2 次元 双峰 × 標準正規 | `u*₁=-2 tanh(...), u*₂=0` |
| `exp6_eb_*_interp.py` (preprocess / alltime / baselines / evaluate) | EB scRNA-seq, $d=30$ | §4.6: 実データでの軌跡推定、補間設定（held-out day 15、All-time vs WOT vs zero） |
| `exp6_eb_bootstrap_stability.py` | EB scRNA-seq, $d=30$ | §4.6: ブートストラップによる速度場の安定性診断（All-time vs WOT） |
| `exp_stochastic_1d_train.py` / `exp_stochastic_1d_metrics.py` | 1 次元 Nelson 問題 `σ=1` | 確率ドリフト `u*=-x/2+1.5+t`（§4.7: 学習 → W₂/MMD 評価） |
| `appA_sensitivity.py` | Exp 1 を基に `(M, N, λ)` 感度解析 | — |
| `appB_dim_scaling.py` | `d = 1…10` Gauss 並進 | 次元スケーリング |

### 実行方法

```bash
pip install -e .                  # alltime_ot をインストール
python experiments/exp1_gaussian.py
python experiments/exp2_roundtrip.py
# ...
```

図と JSON/テーブルは `output/expN/` 以下に出力されます
（`ALLTIME_OT_OUT` 環境変数で変更可能）。

---

## ベースラインの実装について

`alltime_ot/baselines.py` に 2 つの比較手法を集約しています。
これらは RKHS 損失とは別系統で、独自の解析的勾配を持つため
NumPy 実装のままにしてあります（`scipy.interpolate.interp1d` と
`POT` を利用）。

1. **Affine MMOT** (`make_affine_mmot_loss_grad`)
   - `T_k(x) = A_k x + b_k` の学習可能マップ連鎖。
   - 損失は MMD U 統計量 + 運動エネルギー。
   - `scipy.optimize.minimize` で最適化。
2. **Sinkhorn WOT** (`sinkhorn_wot_drift`)
   - 連続するスナップショット間にエントロピック OT（`ot.sinkhorn`）
     を適用し、barycentric projection でドリフトを読み取ります。
   - 1 次元では `position_dependent=True` として `interp1d` を返します。

---

## テスト的に動かす場合

全実験を一気に流すと計算時間がそれなりにかかるため、
動作確認だけなら以下が軽量です。

```bash
python experiments/exp1_gaussian.py        # 数十秒〜1 分程度
python experiments/exp2_flow_matching.py   # 数秒
```

`exp2_mmot_learned_affine.py` や `exp4_baselines.py` はハイパー
パラメータのスイープを含むため分単位の時間がかかります。

---

## 引用

本コードを利用される場合、付随する論文を引用してください：

```bibtex
@article{Nakano2026alltimeot,
  author  = {Nakano, Yumiharu},
  title   = {Continuum-marginal optimal transport: a mesh-free kernel method},
  journal = {arXiv preprint arXiv:2604.24226},
  year    = {2026},
}
```


---

## ライセンス

MIT ライセンス
