# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## コマンド

依存は `uv` で管理している（`uv.lock` はコミット済み）。実行はすべて `uv run` 経由で行う。

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run pytest tests/test_metrics.py::test_nss_is_zero_for_a_flat_prediction
```

学習と、ダウンロード → train → validation → test → 画像ログまでを 2 バッチで通す
ドライラン（1 分程度）:

```bash
uv run illust-salmap --model unet --dataset salicon
```

```bash
uv run illust-salmap --model dummy --dataset salicon --smoke
```

`illust-salmap` は `pyproject.toml` の `[project.scripts]` で定義されたエントリポイント。
`uv run python -m illust_salmap ...` でも同じものが動く。

各モデルモジュールには `__main__` ブロックがあり、`torchinfo` のサマリ出力と
`ez_bench.benchmark` を実行する:

```bash
uv run python -m illust_salmap.models.unet.unet
```

4 つとも非対話で完走する。`unet_v3.py` だけはチェックポイントパスを任意の位置引数で
受け取り、学習済みの重みを読んだ状態でサマリを出せる:

```bash
uv run python -m illust_salmap.models.unet.unet_v3 path/to.ckpt
```

テストはネットワーク不要（データセットのインストーラはフェイクに差し替えられる）。
`tests/test_cli.py` が実際に CPU 学習を 2 回まわすため、全体で 25 秒ほどかかる。

## アーキテクチャ

パイプラインは 4 層で、各層は上位層を知らない:
`installer/`（アーカイブ取得）→ `dataset/`（LightningDataModule）→ `training/`
（LightningModule ラッパー）→ `cli.py`（配線と CLI）。すべて `illust_salmap/` の下にある。

この依存方向は守ること。特に `dataset/` から `training/` を import しない——以前
`calculate_mean_std` を `training/utils.py` から取っていて破れていた。データセット単位の
処理は `dataset/stats.py` に置く。

### cli.py がレジストリであり、その規約はテストで縛られている

`DATASETS` と `MODELS` は CLI 名からコンストラクタへの単なる dict で、そこから導かれる
規約を `tests/test_cli.py` が検証している:

- **モデルを追加する場合**: `models/<系統>/<name>.py`（単発なら `models/<name>.py`）に、素の `nn.Module` を返す
  ファクトリ関数 `def <name>(ckpt_path=None)` を書き、`MODELS` に登録する。ファクトリの
  引数は `ckpt_path` という名前ひとつでなければならない（テストがシグネチャを厳密に
  チェックする）。履歴にある複数の "fix factory function name" コミットは、モジュール名と
  ファクトリ名がずれた件の修正である。登録されているのは UNet 系列 4 種と、`--smoke` と
  `tests/test_cli.py` が使う軽量な `dummy` だけ（lraspp・deeplab・resunet は削除済み）。
- **データセットを追加する場合**: `cli.py` がすべて同一の方法で構築するため、DataModule の
  コンストラクタは `root`、`batch_size`、`num_workers`、`img_size`、`seed` を受け取る必要がある。

### models/ のレイアウト

```
models/
  blocks.py       ← 系統をまたいで共有する部品
  checkpoint.py   ← 汎用（load_weights）
  ez_bench.py     ← 汎用（benchmark）
  dummy_net.py    ← どの系統にも属さない単発モデル
  unet/
    __init__.py   ← docstring のみ。import は置かない（後述）
    unet.py  unet_v2.py  unet_v3.py  unet_lite.py
```

系統ごとにディレクトリを切るが、**`blocks.py` は `models/` 直下に残す。** スキップ融合・
`DecoderBlock`・`UpsampleBlock` は畳み込みエンコーダに固有ではなく、密な予測を出す
ViT 系のモデルも同じデコーダを必要とする（パッチ 16 で 256px を入れるとトークンは 16×16
で、2 倍を 4 回して 256px に戻る——今の UNet のデコーダと同じ段数）。系統に固有なのは
`EncoderBlock` だけだが、分ける相手ができるまでは一緒に置いておく。

**パッケージ内のモジュール名はファクトリ名と一致させること**（`unet_v3.py` が `unet_v3`
を定義する）。`v3.py` のように短縮しないこと——テストはファクトリのシグネチャしか見て
いないのでズレは機械的に検出されず、履歴の "fix factory function name" 連発がまさに
それである。

**`models/unet/__init__.py` に import を置かないこと。** ファクトリを再エクスポートすれば
`cli.py` は 1 行で 4 つ取れるが、そうすると `python -m illust_salmap.models.unet.unet_v3`
がモジュールを 2 回実行する——`__init__.py` が読み込んだ分と、`__main__` として実行される
分。`RuntimeWarning` が出て、クラスオブジェクトが実際に 2 つできる。`__main__` ブロックは
常用のデバッグ経路なので、import 4 行の方を取る。ViT 側のパッケージでも同じ。

### モデルの系譜

`UNetV3` は `UNetV2` をスケール可能にするために書かれたもので、両者は独立した設計では
ない。**`UNetV2` は `UNetV3` のサブクラスで、設定が違うだけ**（4 段の 64/128/256/512、
conv1 がカーネル 5、残差ショートカット無し）。ファクトリで `UNetV3` を返すのではなく
サブクラスにしてあるのは、`training/paths.py:get_save_path` が出力先をネットワークの
クラス名から導くため——`UNetV3` を返すと `--model unet_v2` と `--model unet_v3` が
同じディレクトリに書き込む。`UNet` は SE も残差も持たない別系統。

`UNetV3` のスケールのつまみは `num_blocks`（深さ）と `max_channels`（幅の上限）。
上限が無いとボトルネックが `base_channels * 2**num_blocks` になり、畳み込みのコストは
その 2 乗なので深さが幅に縛られる。既定の `max_channels=512` は既定の設定では no-op:

| num_blocks | 上限あり | 上限なし |
|---|---|---|
| 4 | 13,736,390 | 13,736,390 |
| 5 | 28,719,047 | 71,652,807 |
| 6 | 38,457,288 | 219,395,528 |
| 7 | 53,439,945 | 1,145,873,865 |

ボトルネックだけは `kernel_size` を受け取らずカーネル 3 固定。dilation が既に 16 なので
5x5 では受容野が 65px になり、32px の特徴マップではほとんどパディングを畳み込むだけの
まま、ネットワークで最も幅の広い地点で重みが 2.8 倍になる。

### models/blocks.py

`EncoderBlock`、`DecoderBlock`、`UpsampleBlock`、`SEBlock`、スキップ融合 4 種を持つ。
`UNet`・`UNetV2`・`UNetV3` が共有する。

**1 段の構成は `upsample → skip → DecoderBlock`。** モデル側でこう書く:

```python
dec4 = self.decoder4(self.skip(self.up4(bottle), enc4))
```

先にアップサンプルしてから融合するのは、エンコーダ特徴を**それ自身の解像度のまま**
入れるため。逆順（融合してからアップサンプル）だと、細部が転置畳み込みを通ってから
出てくることになる。加算スキップの場合は解像度の制約から順序が事実上決まる。

**規約: 線形演算（conv / convT）を隣接させない。** BN は推論時にアフィンなので、線形演算が
隣り合うとその区間は畳み込み 1 個分に潰れ、パラメータだけ消費して表現力を足さない。
`DecoderBlock` は conv・BN・活性化を 1 組として 2 回繰り返し、活性化で終わる
（BN で終わらない）。`UpsampleBlock` が転置畳み込みの後に BN と活性化を持っているのは
このためで、これが無いと転置畳み込みと `conv1` の間にはスキップ融合（線形）しか無く、
2 つが 1 つに畳める。**原典の U-Net はまさにその隣接を持っている**（up-conv → concat →
3x3 conv の間に非線形性が無い）が、ここでは持たせない。

この規約が守れているかは合成性で機械的に検査できる:
`f(a) + f(b) - f(a+b) - f(0)` が 0 ならその区間に非線形性は一切ない。ブロック単体だけで
なく、`upsample → skip → decoder` の区間、およびブロックを跨いだ区間でも 0 にならない
ことを確認すること（`AddSkip` は `GatedSkip` と違って活性化を持たないため、ここが
0 になりうる）。

**エンコーダブロックはダウンサンプルしない。** プーリング／stride 2 conv はモデルが
ブロックの**間**で行い、ブロックの出力がそのまま skip の分岐点になる。こうしないと
skip がダウンサンプル後のテンソルになり、最も細かい skip が入力解像度の半分になって
しまう。以前はそうなっていて、256px の出力にエンコーダ特徴が一切入らず、最後の
128→256 が転置畳み込みだけで作られていた（実質「128px の U-Net + 学習可能な 2 倍
アップサンプル」）。UNetV3 では `downsamples` / `upsamples` を `ModuleList` で持ち、
スケールしない段は `Identity` を入れてリストの添字を `encoders` / `decoders` と
揃えている。

**`EncoderBlock` も共有している。** `conv → BN → act → [SE] → conv → BN → act →
[+shortcut] → dropout` で、`se` と `residual` の 2 つが `UNet`（どちらも無し）と
UNetV2 の設定（SE のみ）と `UNetV3`（両方）を分ける。`kernel_size` と `dilation` は
ふつうの引数。conv1 のパディングは `"same"`——`UNet` の `padding=2`（カーネル 5、
dilation 1）と数値的に同一だが、固定パディングは dilation を動かした瞬間に黙って
ずれる。

**ボトルネックも `EncoderBlock`。** 専用クラスは無い。`UNet` は
`dropout_prob=0.0, se=False, residual=False`、`UNetV3` は `dilation=16`。どちらも
中身は `conv → BN → act` を 2 回で、専用クラスだった頃と数値的に一致する。
`dropout_prob=0` のとき `Dropout2d` は作られない（`se_block` や `shortcut` と同じ
「使うときだけ作る」扱い。`Dropout2d(p=0)` は死んだレイヤ）。

`UNetLite` はエンコーダもデコーダもボトルネックも共有していない。conv 1 本の軽量
ブロックがその存在理由で、共有ブロックにすると倍になる（`UNetLite` の `Bottleneck` は
さらに stride 2 でダウンサンプルもするので、そもそも役割が違う）。

**`UNetLite` は skip の分岐点だけ未修正。** デコーダのアフィン問題は解消済み
（`Decoder` が活性化で終わるようにした。以前は `Conv2d → BatchNorm2d` で終わり、
`AddSkip` も活性化を持たないため次のブロックの `ConvTranspose2d` までがアフィンだった）
が、`Encoder` は `MaxPool2d` を内蔵したままなので skip はプール後の解像度で、最も
細かい skip は 128px にとどまる。`Bottleneck` が stride 2 で 6 段目を作るのに
エンコーダ出力は 5 つしかなく、`bottle_512` を自分自身への skip として使い回している
——ここを直すには段数の設計から見直しが要る。

UNetV3 のエンコーダ側にはもう 1 箇所隣接が残っている: 外に出した stride 2 の
`downsample` conv が、次のブロックの `conv1` に素通しで入る。修正前から同じで、
今回の範囲外。

**共有ブロックはモデルを結合させる。** ここへの変更は `DecoderBlock` を使う全モデルの
ベースラインを動かし、過去のログが比較不能になる（`*_scc` と `*_cc` で起きたのと同じ
事故）。共有部品を触るときは、どのモデルの数値が無効になるかを意識すること。

### スキップ接続はデコーダの外

スキップの融合は `models/blocks.py`（`ConcatSkip`・`AddSkip`・`GatedSkip`・`NoSkip`）にあり、
DecoderBlock は持たない。配線するのはモデル側で、`decoder(skip(x, encoder_output))` と書く。
どのデコーダも融合を forward の 1 行目で済ませてエンコーダ特徴を二度と見ていなかったので、
ブロックの内部処理とは元から絡んでいなかった。外に出したことで `DecoderBlock.forward` は
`(Tensor) -> Tensor` の普通の Module に戻り、concat・加算・ゲート・無しの選択が
デコーダの実装違いではなくモデルの配線の違いになる。

- `ConcatSkip` はチャネルが増えるため、下流の DecoderBlock には**融合後の幅**を渡す。
  `UNet` の `DecoderBlock(1024, 256)` の 1024 は「bottleneck の 512 + encoder4 の 512」。
  以前はこれを DecoderBlock 内の `in_channels * 2` が暗黙に吸収していた。
- `AddSkip`・`GatedSkip` はチャネル数を保つので x と y の幅が一致している必要がある。
  `UNetV3` がスキップを upsample より**前**に入れているのはこのためで、その地点でのみ
  エンコーダ出力がチャネル・解像度とも一致する。
- 解像度が合わない場合は既定で `ValueError`。加算は空間的に小さいスキップを
  ブロードキャストして黙って通してしまう（8x8 に対する 1x1 は合法）ため、
  リサイズが必要なら `resize=True` を明示する（`UNetLite` のみ該当）。
- `GatedSkip` は `sigmoid(skip_gate) * y`。ゲートは倍率なので 0 まで閉じられ、
  既定の `skip_weight=0.5` では `0.6225 * y`。以前は `sigmoid(skip_gate * y)` と
  sigmoid の**内側**に掛けており、ゲートが何を学習しても寄与は (0, 1) に制限されて
  常に正、`y = 0` の画素では 0 ではなく 0.5 を足していた（閉じないゲート）。

### データセット

各データセットは 2 クラス構成。`XxxDataset(Dataset)` がアーカイブをインストールして
`(画像パス, マップパス)` のペアをキャッシュし、`Xxx(LightningDataModule)` が分割して配信する。

- `Dataset` 側は `pair_categories`（サンプルごとの、アーカイブ本来のカテゴリ）も保持する。
  `dataset/split.py:stratified_indices` がこれを使い、分割間でカテゴリ構成比を保つ。
  シードはカテゴリごとに `hash_group` で決まる（`hash()` はプロセスごとにソルトされるため
  使えない）ので、分割は再現可能。
- `setup()` は `self.train is not None` なら早期 return する。Lightning は stage ごとに
  これを呼ぶため、このガードがないと `fit` と `test` が別々の分割を見てしまう。
- **3 つのデータセットすべてが `test_dataloader` で validation 分割を返す。** CAT2000 の
  test の視線データは非公開で、SALICON の再パッケージ版はフォルダ構成が未検証のため。
  よってログ上の `test_*` メトリクスはホールドアウトした validation の数値であり、
  ベンチマークスコアではない。
- 変換は画像・マップとも [-1, 1] に正規化する（`Normalize([0.5], [0.5])`）。マップは
  1 チャネルのグレースケールで、モデル側の出力も 1 チャネルで合わせる。
- **ヘッドは `Tanh`。** 過去に `Sigmoid` で学習が進まなかったという実測があり、原因は
  ターゲットのレンジとの不一致である。サリエンシーマップは 9 割以上の画素が背景 = -1
  付近だが、`Sigmoid` の出力は (0, 1) なのでそこに到達できない。しかも 0 に近づけるには
  pre-activation を大きな負値へ飽和させるしかなく、そこでは勾配が消える。実測:

  | ヘッド | MSE 到達値 | 理論下限 | 勾配の減衰 |
  |---|---|---|---|
  | Sigmoid | 0.8252 | 0.8146 | ×0.016 |
  | Tanh | 0.0006 | 0.0 | 収束 |

  つまり下限に貼り付いたまま勾配が死ぬ。`--loss saliency` の各項はスケール・オフセット
  不変なのでこの制約は掛からない（`Sigmoid` でも学習する）が、`--loss mse` では致命的
  なので `Tanh` に統一してある。
- ペアリングは `dataset/pairing.py:paired_paths` が stem 突き合わせで行い、片方に無い
  stem があれば `FileNotFoundError` を投げる。以前は 2 つの glob 結果を位置で `zip` して
  いたため、マップが 1 枚欠けるとそれ以降の全ペアが 1 つずつずれ、しかも無言だった。

### installer/

`DatasetInstaller` は `Downloader`（HTTP または Google Drive）と `Extractor` を組み合わせ、
両者があえて持たない責務——リトライ、壊れたアーカイブの再取得、展開先を記録する
アーカイブ単位のマーカー `.{archive}.installed.json`——を引き受ける。

`install()` はコンテンツのディレクトリを返す。パスを自前で組み立てず、この戻り値を使うこと。
`ZipExtractor` はトップレベルのディレクトリが 1 つだけならそれを剥がし、そうでなければ
`dest_root/<アーカイブの stem>` に展開するため、レイアウトはアーカイブ次第で変わる。

### training/

- ラッパーは `SaliencyModel` ひとつだけで、ネットワークが単一のテンソルを返すことを
  前提にしている。現存するモデルはすべてこれを満たす。
- `(out, aux)` を返すネットワーク向けの `MultiOutSaliencyModel`、未完成の GAN ラッパー
  `salency_gan_model.py`、Discord 通知コールバックは、いずれも未使用だったため削除した。
  補助出力を持つモデルを追加するなら（`SaliencyModel` はタプルをそのまま損失関数に渡して
  落ちる）、`MultiOutSaliencyModel` を git 履歴から戻すのが早い。
- `visualize.py:generate_plot` は pyplot ではなく Figure と Agg キャンバスを直接使う。バッファに
  書き出すだけなのに pyplot を通すと設定中の対話バックエンド（Windows なら TkAgg）に回され、
  学習中に GUI ウィンドウが開いたり、ヘッドレス環境や Tk が壊れた環境では例外になるため。
  pyplot のグローバルな figure レジストリも経由しないので、epoch をまたいで溜まらない。
- `metrics.py` はメトリクス一式（`SaliencyMetrics`、stage ごとに 1 インスタンス）と、
  torchmetrics に相当物がない `SIM`・`NSS`・`CC`・`AUC` を持つ。後ろ 2 つは torchmetrics
  に同名のものがあるが別物なので自前で持っている:
  - `SpatialCorrelationCoefficient` は名前に反して高域通過フィルタ付きの局所窓相関で、
    サリエンシー文献の CC（大域 Pearson 相関）ではない。GT に微小な画素ノイズを載せた
    予測が CC 0.98 に対し 0.01 と出る。**`scc` という名前でログしていたのはこれなので、
    過去のログの `*_scc` は現在の `*_cc` と比較できない。**
  - `torchmetrics.AUROC` は渡されたものを 1 つのランキングにプールするため、`(B, N)` を
    渡すとバッチ全体を 1 枚の画像として採点してしまう。サリエンシーの AUC は画像単位。
    通常のデータでの差は 1〜5 ポイントだが、注視領域の広さが画像間で大きく違うと
    （CAT2000 のカテゴリ間など）大きく開く。
- `convert_*` ヘルパは取り違えやすい規約を表現している: `convert_kl_div` は
  `D_KL(gt || pred)` のため `(ground truth, prediction)` の順で返し、呼び出し側で
  アンパックが必要。`convert_auroc` は ground truth だけを閾値処理し、予測は連続値の
  まま残す。`normalized()` は画像ごとに独立して min-max する——サリエンシーのメトリクスは
  画像単位で定義されるため、バッチ全体の統計を使うと誤りになる。
- `normalized()`・`to_distribution()`・`correlation_coefficient()` は微分可能で、
  `losses.py` と共有している。指標と最適化対象がずれないようにするためなので、
  `@torch.no_grad()` を付け直さないこと。代わりに各 Metric の `update()` 側で
  勾配を切っている。
- `losses.py:SaliencyLoss`（KL + CC + NSS）が既定の損失。MSE はサリエンシーマップの
  大半が背景であるためデータセット平均のマップに引かれ、報告している指標のどれも
  最適化しない。`--loss mse` で従来の挙動に戻せる。重み（既定 1.0 / 1.0 / 0.1）は
  出発点であって調整済みの値ではない。
- 損失の各項は予測のスケールとオフセットに対して不変（指標と同じ性質）。よって
  出力の絶対レンジは損失から一切拘束されない。
- `cli.py:MONITORS` がチェックポイント選択と早期終了の対象を定義する。`--monitor` は
  ここのキーに限られ、min/max の向きも同じ dict が持つ。`val_loss` は `--loss` が
  選んだものなので、`--loss mse` のときに `val_loss` を監視すると「MSE が良いモデル」を
  選ぶことになる。

### 出力とチェックポイント

学習結果は `<output-root>/<DataModuleのクラス名>/<ネットワークのクラス名>/{logs,checkpoints}`
に書かれる（`training/paths.py:get_save_path`）。パスがクラス名から導かれるので、クラスを
リネームすると出力先が黙って変わる。

`models/checkpoint.py:load_weights` は Lightning のラッパーが付ける `model.` プレフィックスを
剥がす。プレフィックス付きの解釈を先に試し、次にチェックポイントをそのまま試す。これにより
推論やエクスポートが `illust_salmap.training`・pytorch-lightning・torchmetrics に一切依存しない。
これが `--weights`（ネットワークの重みのみ）の実体で、`--resume` は別経路で
`trainer.fit(ckpt_path=...)` にそのまま渡され、optimizer と epoch を復元する。

## 環境

`torch` / `torchvision` は明示的な PyTorch インデックス経由でプラットフォームごとに解決される。
Windows（開発機）は CPU ホイール、Linux（学習機）は CUDA 12.4。torch 2.5.1 + cu124 は sm_90 までの
対応なので、Blackwell 世代（5090、B200）には torch 2.7+ / cu128 が必要になる。

RunPod ではポッド停止時にコンテナディスクが消えるため、`uv sync` の前に uv の参照先を
`/workspace` ボリュームに向ける:

```bash
export UV_CACHE_DIR=/workspace/.uv-cache UV_PROJECT_ENVIRONMENT=/workspace/.venv UV_LINK_MODE=copy
```

`scripts/runpod_setup.sh` がこれを含めた立ち上げ（uv 導入・sm チェック・`uv sync`・
スモークラン）を一括で行う。冪等なのでポッド再開のたびに実行してよい。
