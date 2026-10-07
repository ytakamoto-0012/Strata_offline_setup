# Strata をオフラインPCにセットアップする（手動収集版）

インターネットにつながるPC（以下「オンラインPC」）で必要なファイルを手で集め、外付けディスクなどで
インターネットにつながらないPC（以下「オフラインPC」）へ運んでセットアップする手順です。

- 対象バージョン: **Strata v0.1.40.2**（2026-10-07 公開の最新 release）。ソース・エンジン・wheel・モデルの
  固定リビジョンはすべてこのタグの `setup.py` に合わせてあります。別のバージョンを使うときは
  [9. 別のバージョンにするとき](#9-別のバージョンにするとき) を見てください。
- 対象 OS / GPU: **Windows 10/11 + NVIDIA**。Linux 用のエンジンは release に無く、その場でのビルド（CUDA Toolkit
  とコンパイラが必要）になるため、この手順の対象外です。
- 検証状況: v0.1.40.2 の `setup.py` を読んで組み立て、2026-10-07 に各 URL・ファイルサイズ・SHA-256・wheel の
  依存関係を確認しました。オフラインPCでの通しの実行はまだしていません。

---

## 0. 全体の流れ

```
オンラインPC                                   オフラインPC
-------------------------------------------    --------------------------------------------------
1. ファイルを集める（ソース、Python、wheel、     4. ドライバをインストール、Python フォルダを配置
   llama.cpp、エンジン、モデル、MTP）             5. ソースを展開し .venv と pip.ini を作る
2. SHA-256 を確かめる                          6. llama.cpp・モデル・MTP を所定の場所に置き .done を作る
3. 外付けディスクにまとめる          ──運ぶ──>   7. setup をオフラインで実行
                                               8. 自動チューニング（--calibrate）→ 起動
```

setup.py がネットから取るものと、オフラインでの渡し方の対応です。

| setup.py が取るもの | 通常の取得元 | オフラインでの渡し方 |
|---|---|---|
| Python 3.12 | winget / python.org | インストール済みフォルダをコピーして配置（PATH 不要） |
| Python パッケージ（requirements.txt）と CUDA ライブラリ | PyPI | wheel フォルダ + `.venv\pip.ini` |
| llama.cpp ソース zip | GitHub | `third_party\` に zip と `.done` を置く |
| エンジン zip | GitHub release | `--prebuilt <フォルダ>` |
| モデル GGUF | Hugging Face | `Strata-data\models\<サイズ>\` に置く |
| 画像エンコーダ（mmproj） | Hugging Face | `Strata-data\models\` に置き `.done` を作る |
| MTP ドラフト層（約 6.5 GB） | Qwen の元チェックポイント | `Strata-data\mtp\` ごとコピー |

起動後のサーバーは、画像を URL で渡されたとき以外はネットに出ません。

---

## 1. PC ごとのモデルと設定

| | タイプA | タイプB | タイプC |
|---|---|---|---|
| RAM | 64 GB | 256 GB | 256 GB |
| GPU | RTX PRO 4000 Blackwell 24 GB | RTX 2000 Ada 16 GB | RTX A2000 12 GB（※） |
| GPU 世代（compute capability） | sm_120 | sm_89 | sm_86 |
| エンジン | `strata-windows-x64.zip` | 同左 | 同左 |
| モデル（`--model`） | **IQ2_XS** | **IQ3_S** | **IQ3_S** |
| コンテキスト（`--context`） | **204800**（200K） | 同左 | 同左 |
| 画像 | **使う**（`--vision yes`、GPU で処理） | 同左 | 同左 |

※ RTX 2000 Ada は 16 GB の製品しか無いので、12 GB の機種は Ampere 世代の **RTX A2000 12GB** と想定しています。
どちらの世代でも v0.1.40.2 のエンジン（対応: sm_75 / 86 / 89 / 120）で動きます。実機では次のコマンドで確認してください。

```bat
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
```

選んだ理由:

- **モデル**: [docs/MODELS.md](MODELS.md) の「Pick by RAM」の推奨に合わせています。
  - 64 GB は IQ2_XS を推奨（IQ3_XXS / IQ3_S も入りますが、IQ3_S は他のソフトをほとんど開かない前提）。
    `--yes` だけにすると、setup は RAM 60 GB 以上で IQ3_XXS を選ぶので、`--model` は必ず指定してください。
  - 96 GB 以上は IQ3_S（公開ベンチマークで元のモデルと同等の品質）。
  - Unsloth の UD-IQ4_XS（94 GB）も 256 GB なら選べますが、この手順では扱いません。
- **コンテキスト**: 必要条件が「160000 以上」なので、それを満たす setup の選択肢の **204800（200K）** にします。
  - モデルの学習時の長さ 262144 より短いので、RoPE スケーリングは入りません。262144 を超えると、実験的な拡張が入ります。
  - setup の推奨値（VRAM が 14 GB 未満なら 32K、20 GB 未満なら 64K、それ以上なら 128K）より長いので、
    `--context` で必ず指定します。指定した値はそのまま使われます。
  - `--context 163840` のように、選択肢に無い値も受け付けます。
- **長いコンテキストと VRAM**: 64K 以上では、setup が KV ストリーミングを有効にします。
  - KV キャッシュの本体を RAM に置き、VRAM には注意機構が読む部分（各層 32K 位置分）だけを置く方式です。
    そのため、コンテキストを長くしても VRAM の使用はほとんど増えません。
  - RAM は、8-bit KV の 200K で約 2.8 GB 増えます。
  - setup が有効にする条件は「RAM ≧ モデルの必要 RAM + KV の RAM + 1 GB」です。どのタイプも満たします。
    - タイプA（IQ2_XS）: 48 + 2.8 + 1 = 51.8 GB ≦ 64 GB
    - タイプB・C（IQ3_S）: 62 + 2.8 + 1 = 65.8 GB ≦ 256 GB
  - IQ3_S の RAM の見積もり（エキスパート 50.3 GB + KV 2.8 GB + 余裕 24 GB = 約 77 GB）も 256 GB に収まるので、
    setup は警告を出しません。
- **画像**: 画像エンコーダ（mmproj、0.9 GB）を集めて `--vision yes` にします。
  - エンコーダは GPU で動きます。エンジン zip の画像エンコーダは sm_86 / 89 / 120 に対応しています。
  - 画像エンコーダ用に VRAM を約 1.4 GB 空けておくため、GPU に載るエキスパートが減り、文章の出力は数 % 遅くなります
    （setup の説明）。

参考になる実測値（[docs/DETAILS.md](DETAILS.md)）:

- RTX 5070（12 GB）+ Ryzen 5 7600 + 64 GB RAM、IQ2_XS、262K のプロンプト（エンジン 0.1.22）で、
  - 出力 52.8 tokens/s
  - プロンプトの読み込み 1,181 tokens/s（画像をオンにした状態で測った値）
- IQ3_S の 128K を超える長さは、プロジェクトでは測っていません（64 GB の PC で 256K を動かした利用者の報告はあります。#406）。
- 上の3タイプの PC では、どれも測っていません。

---

## 2. オンラインPCで集める

### 2.0 用意するもの

- Windows 10/11 の PC（GPU は不要）
- **Python 3.12**（オフラインPCと同じマイナーバージョン。wheel が `cp312` 用になるため）。以下では `C:\Python312`
  にあるものとして、フルパスで呼びます（2.2 を参照）
- curl（Windows 10/11 に標準で入っています）
- 外付けディスク: IQ2_XS と IQ3_S の両方と画像エンコーダを集めると約 **136 GB**

以下では、外付けディスクを `E:`、作業フォルダを `E:\StrataOffline` とします。

### 2.1 フォルダ構成（最終形）

```
E:\StrataOffline\
  01_python\   Python312\（インストール済みの Python フォルダ一式）
  02_driver\   NVIDIA ドライバ（GPU ごと。580 以上）
  03_source\   Strata-0.1.40.2.zip
  04_llama\    llama.cpp-3cf0325.zip
  05_engine\   strata-windows-x64.zip
  06_wheels\   *.whl（18 個）
  07_models\
     mmproj-Qwen3.8-Flash-Next-BF16.gguf   （画像エンコーダ。全タイプ共通）
     IQ2_XS\   Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
               Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
     IQ3_S\    Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
               Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  08_mtp\
     mtp\      （MTP ドラフト層一式。rt\experts.bin を含む）
```

### 2.2 Python（3.12、コピー配置用のフォルダ）

オフラインPCにはインストーラーを使わず、**インストール済みの Python フォルダをコピーして配置**します（PATH も
py ランチャーも使いません）。オンラインPCで、インストーラーを使って決まったフォルダに入れ、そのフォルダを運びます。
バージョンは START-HERE.bat が自動で入れるものと同じ 3.12.10 です。

```bat
curl -L -o %TEMP%\python-3.12.10-amd64.exe https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
%TEMP%\python-3.12.10-amd64.exe /quiet InstallAllUsers=0 TargetDir=C:\Python312 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0 AssociateFiles=0
C:\Python312\python.exe --version
robocopy C:\Python312 E:\StrataOffline\01_python\Python312 /E
```

- オンラインPCに 3.12.10 がすでに入っていると、インストーラーは新しいフォルダに入れずに「変更」の画面になります。
  その場合は、入っているフォルダ（ユーザー単位のインストールなら `%LOCALAPPDATA%\Programs\Python\Python312`）を
  そのままコピーして構いません。その場合は `C:\Python312` にもコピーしておくと、以下のコマンドをそのまま使えます
  （コピーした Python は PATH なしで動きます。5 を参照）。
- フォルダには `python.exe`、`python312.dll`、`vcruntime140.dll`、`Lib\`、`DLLs\` が揃っていて、それだけで動きます。
  .venv は独立しているので、`Lib\site-packages` に別のパッケージが入っていても影響しません。
- 以下の手順では、オンラインPCの Python も `C:\Python312\python.exe` とフルパスで呼びます。

### 2.3 NVIDIA ドライバ

<https://www.nvidia.com/drivers> から GPU ごとにドライバを落とし、`02_driver\` に入れます。
ドライバは **580 以上** が必要です（エンジンが CUDA 13.0 でビルドされているため。setup はこれ未満だと止まります）。
RTX PRO / RTX Ada / RTX A シリーズは、それぞれの製品を選んで落としてください。

### 2.4 Strata のソース（v0.1.40.2）

```bat
mkdir E:\StrataOffline\03_source
curl -L -o E:\StrataOffline\03_source\Strata-0.1.40.2.zip https://github.com/Niko1221/Strata/archive/refs/tags/v0.1.40.2.zip
```

`git clone -b v0.1.40.2 https://github.com/Niko1221/Strata.git` でも同じです。手元の古いチェックアウトは使わないで
ください。エンジンの版（`MIN_ENGINE`）や wheel の固定版がタグと合わなくなります。

### 2.5 llama.cpp のソース zip（commit 3cf0325）

```bat
mkdir E:\StrataOffline\04_llama
curl -L -o E:\StrataOffline\04_llama\llama.cpp-3cf0325.zip https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.zip
```

保存名は **`llama.cpp-3cf0325.zip`**（commit の先頭7文字）にしてください。setup.py はこの名前で探します。

### 2.6 エンジン

```bat
mkdir E:\StrataOffline\05_engine
curl -L -o E:\StrataOffline\05_engine\strata-windows-x64.zip https://github.com/Niko1221/Strata/releases/download/v0.1.40.2/strata-windows-x64.zip
```

中身は `strata.exe`、`strata-vision.exe`、`BUILD.json`（version 0.1.40.2、archs 75 / 86 / 89 / 120、CUDA 13.0）です。
3タイプとも同じ zip で動きます。

### 2.7 Python パッケージ（wheel）

オンラインPCの Python 3.12 で、Windows 64bit・Python 3.12 用の wheel を落とします。

```bat
mkdir E:\StrataOffline\06_wheels
cd /d E:\StrataOffline
powershell -Command "Expand-Archive 03_source\Strata-0.1.40.2.zip -DestinationPath C:\work"
C:\Python312\python.exe -m pip download -d E:\StrataOffline\06_wheels --only-binary=:all: --platform win_amd64 --python-version 3.12 --implementation cp -r C:\work\Strata-0.1.40.2\requirements.txt nvidia-cublas==13.0.2.14 nvidia-cuda-runtime==13.0.96 colorama==0.4.6
```

次の **18 ファイル** ができれば揃っています（2026-10-07 に pip の依存解決で確認）。

```
nvidia_cublas-13.0.2.14-py3-none-win_amd64.whl        nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl
numpy-2.5.3-cp312-cp312-win_amd64.whl                 jinja2-3.1.6-py3-none-any.whl
regex-2026.9.10-cp312-cp312-win_amd64.whl             pyyaml-6.0.3-cp312-cp312-win_amd64.whl
tqdm-4.70.1-py3-none-any.whl                          requests-2.34.2-py3-none-any.whl
cmake-4.4.3-py3-none-win_amd64.whl                    ninja-1.13.2-py3-none-win_amd64.whl
pillow-12.3.0-cp312-cp312-win_amd64.whl               psutil-7.2.2-cp37-abi3-win_amd64.whl
markupsafe-3.0.3-cp312-cp312-win_amd64.whl            certifi-2026.7.22-py3-none-any.whl
charset_normalizer-3.5.1-cp312-cp312-win_amd64.whl    idna-3.20-py3-none-any.whl
urllib3-2.8.0-py3-none-any.whl                        colorama-0.4.6-py2.py3-none-any.whl
```

`nvidia-cublas` と `nvidia-cuda-runtime` は、release のエンジンが読み込む CUDA 13 のライブラリです（約 0.4 GB）。

### 2.8 モデルファイル

ブラウザでも curl でも構いません。URL の `ed59f92…` は setup.py が固定しているリビジョンです。必ずこのリビジョンから
落としてください（`main` からではなく）。

共通の前置き（以下 `%R%`）:

```bat
set R=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09
```

**タイプA 用（IQ2_XS）**

```bat
mkdir E:\StrataOffline\07_models\IQ2_XS
cd /d E:\StrataOffline\07_models\IQ2_XS
curl -L -C - -O %R%/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
curl -L -C - -O %R%/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
```

**タイプB・C 用（IQ3_S）**

```bat
mkdir E:\StrataOffline\07_models\IQ3_S
cd /d E:\StrataOffline\07_models\IQ3_S
curl -L -C - -O %R%/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
copy E:\StrataOffline\07_models\IQ2_XS\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
```

**画像エンコーダ（全タイプ共通）**

```bat
cd /d E:\StrataOffline\07_models
curl -L -C - -O %R%/mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

**shard 2 は、どのサイズでも中身が同じファイル**です（SHA-256 が一致。setup.py も同一ファイルとして扱います）。
IQ2_XS の shard 2 をコピーして名前を変えれば、もう一度落とす必要はありません。`-C -` を付けると、途中で切れた
ダウンロードを続きから再開できます。

| ファイル | バイト数 | SHA-256 |
|---|---:|---|
| `IQ2_XS/…-IQ2_XS-00001-of-00002.gguf` | 39,225,954,592 | `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7` |
| `IQ3_S/…-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `…-00002-of-00002.gguf`（全サイズ共通） | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf`（画像エンコーダ） | 907,543,008 | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |
| （参考）`IQ3_XXS/…-IQ3_XXS-00001-of-00002.gguf` | 47,039,860,096 | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |

### 2.9 MTP ドラフト層

MTP ドラフト層は投機的デコード用の小さな層で、出力を約2倍速くします。setup.py は Qwen の元チェックポイント（360 GB）から、
HTTP の範囲指定で MTP の部分（約 5 GB）だけを読み出して作ります。オフラインPCではこれができないので、オンラインPCで
作ってフォルダごと運びます。**モデルのサイズに関係なく1つで共通**です。

**方法1（手早い）: すでに Strata が入っている PC からコピーする**

Strata をセットアップ済みの PC があれば、その `Strata-data\mtp\` フォルダ（`rt\experts.bin` を含む約 6.5 GB）を
そのまま `E:\StrataOffline\08_mtp\mtp\` にコピーします。この層を作るツール（`mtp_pack.py` と `mtp_rt.py`）は、
v0.1.40 から v0.1.40.2 まで変わっていません。

**方法2: オンラインPCで作る**（GPU は不要。CPU と Python だけで動きます）

```bat
:: 2.7 で展開したソースを使う
cd /d C:\work\Strata-0.1.40.2
C:\Python312\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install --no-index --find-links E:\StrataOffline\06_wheels -r requirements.txt

:: llama.cpp の gguf-py を使えるようにする（展開先はパスの短い場所に。長いパスで失敗することがある）
powershell -Command "Expand-Archive E:\StrataOffline\04_llama\llama.cpp-3cf0325.zip -DestinationPath C:\w"
set STRATA_GGUF_PY=C:\w\llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d\gguf-py

set M=E:\StrataOffline\08_mtp\mtp
.venv\Scripts\python.exe tools\mtp_fetch.py fetch  --out %M%
.venv\Scripts\python.exe tools\mtp_pack.py  --src %M% --experts q2_0 --out %M%\mtp-q2_0.gguf
.venv\Scripts\python.exe tools\mtp_rt.py    --gguf %M%\mtp-q2_0.gguf --out %M%\rt
.venv\Scripts\python.exe tools\mtp_fetch.py verify --out %M%
```

これは setup.py がステップ6で実行するのと同じ3つのコマンドです。最後の `verify` が終了コード 0 で終われば、取得した
テンソルは固定リビジョンの SHA-256 と一致しています（終了コード 3 は壊れたテンソルがあるという意味）。
`fetch` は途中で止まっても、もう一度実行すれば続きから再開します。

できあがった `mtp\` フォルダには、`tensors\`、`mtp-manifest.json`、`mtp-q2_0.gguf`、`rt\experts.bin` などが入っています。
**`tensors\` と `mtp-manifest.json` も消さずに運んでください**。オフラインPCの setup がこれを使って SHA-256 を照合します。

---

## 3. SHA-256 を確かめる

運ぶ前にオンラインPCで確認しておきます。オフラインPCでは、`.done` の付いたファイルは中身を調べずに使われます。

```powershell
Get-FileHash -Algorithm SHA256 E:\StrataOffline\05_engine\strata-windows-x64.zip
Get-FileHash -Algorithm SHA256 E:\StrataOffline\07_models\IQ2_XS\*.gguf
Get-FileHash -Algorithm SHA256 E:\StrataOffline\07_models\IQ3_S\*.gguf
Get-FileHash -Algorithm SHA256 E:\StrataOffline\07_models\mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

（コマンドプロンプトなら `certutil -hashfile <ファイル> SHA256`）

| ファイル | バイト数 | SHA-256（GitHub が公開している値） |
|---|---:|---|
| `strata-windows-x64.zip`（v0.1.40.2） | 135,496,886 | `02901f0cd0691ab33ec827e354ffa72e1ab4f19a093f0280c6deb7327b26eb30` |

モデルの値は [2.8](#28-モデルファイル) の表のとおりです。

エンジンについて補足します。setup.py は通常、エンジン zip を GitHub の API が公開する SHA-256 と照合します。
`--prebuilt` にローカルフォルダを指定すると照合先が無いため、**警告を出して照合せずに入れます**。そのため、
ここで手で確かめておくことが大事です。

---

## 4. オフラインPC: ドライバと Python

1. **NVIDIA ドライバ**（`02_driver\`）をインストールして再起動します。`nvidia-smi` を実行し、ドライバが 580 以上で、
   GPU が表示されることを確認します。
2. **Python 3.12** のフォルダをコピーして配置します。インストーラーも PATH の設定も要りません。

   ```bat
   robocopy E:\StrataOffline\01_python\Python312 C:\Python312 /E
   C:\Python312\python.exe --version
   :: → Python 3.12.10
   ```

   **配置したあとで `C:\Python312` を移動したり名前を変えたりしないでください**。5 で作る .venv は、元の Python の
   場所を絶対パスで記録します（`.venv\pyvenv.cfg` の `home = C:\Python312`）。移動すると .venv が動かなくなります。
   移動した場合は、.venv を消して 5 からやり直します。
3. Windows の仮想メモリ（ページファイル）は「システム管理サイズ」にしておきます。4 GB 未満だと setup が警告を出し、
   モデルが起動しないことがあります。

---

## 5. オフラインPC: ソース・.venv・pip.ini

以下では、Strata を `C:\Strata`、データを `C:\Strata-data`（setup.py の既定は「Strata フォルダの隣の
`Strata-data`」）、運んだファイルを `C:\StrataOffline` に置くとします。

**データフォルダは SSD に置いてください**。HDD だと、29 GB の表をランダムに読むときにプロンプトの処理が数分止まる
ことがあります。

```bat
:: 運んだファイルをローカルにコピー（wheel は後の更新でも使うので残しておく）
robocopy E:\StrataOffline C:\StrataOffline /E

:: ソースを展開して C:\Strata にする（パスが短いほど安全）
powershell -Command "Expand-Archive C:\StrataOffline\03_source\Strata-0.1.40.2.zip -DestinationPath C:\"
ren C:\Strata-0.1.40.2 Strata

:: 配置した Python 3.12 をフルパスで呼んで .venv を作る
C:\Python312\python.exe -m venv C:\Strata\.venv
```

**PATH を通す必要はありません**。理由は次のとおりです。

- START-HERE.bat は、`.venv\Scripts\python.exe` があればそれを直接実行します。PATH から Python を探すのは .venv が
  無いときだけです。
- setup.py・サーバー・チューニング・モデル準備のツールは、子プロセスもすべて自分を動かしている Python
  （`.venv\Scripts\python.exe`）で起動します。setup が書く起動スクリプトにも、その絶対パスが入ります。
- .venv の `python.exe` は、`.venv\pyvenv.cfg` に記録された元の Python（`home = C:\Python312`）を直接使います。

Python の場所が必要なのは .venv を作るこの1回だけで、それもフルパスで渡します。

確認したこと: 2026-10-07 に、インストール済みの Python 3.12.10 のフォルダを別の場所にコピーしました。PATH に Python が
無い状態（`where python` で見つからない状態）で、そこから .venv を作り、.venv の Python で `ssl`・`ctypes` の
読み込みと `pip --version` が動くことを確かめました。

**.venv を作る前に START-HERE.bat を実行しないでください**。.venv が無いと、START-HERE.bat は PATH から Python を探し、
見つからなければ winget や python.org からのインストールを試みて止まります。

次に、**`C:\Strata\.venv\pip.ini`** をメモ帳で作り、以下を書きます。

```ini
[global]
no-index = true
find-links = C:\StrataOffline\06_wheels
```

これで、この .venv の pip は PyPI に行かず、wheel フォルダだけからインストールします。PC 全体の pip 設定は変わりません。
確認:

```bat
C:\Strata\.venv\Scripts\python.exe -m pip config list
:: global.find-links='C:\\StrataOffline\\06_wheels' と global.no-index='true' が出れば OK
```

---

## 6. オフラインPC: ファイルを所定の場所に置く

### 6.1 llama.cpp

```bat
copy C:\StrataOffline\04_llama\llama.cpp-3cf0325.zip C:\Strata\third_party\
type nul > C:\Strata\third_party\llama.cpp-3cf0325.zip.done
```

`.done` が無いと、setup.py は GitHub から落とし直そうとして止まります。setup はこの zip を
`third_party\llama.cpp\` に展開したあと、zip と `.done` を消します。

### 6.2 モデル

モデルのフォルダ名はサイズ名そのもの（Qwen 本体の場合）です。

**タイプA**

```bat
mkdir C:\Strata-data\models\IQ2_XS
robocopy C:\StrataOffline\07_models\IQ2_XS C:\Strata-data\models\IQ2_XS *.gguf
```

**タイプB・C**

```bat
mkdir C:\Strata-data\models\IQ3_S
robocopy C:\StrataOffline\07_models\IQ3_S C:\Strata-data\models\IQ3_S *.gguf
```

モデルの shard には `.done` が無くても構いません。setup.py は、手でコピーされた shard を中のテンソル目録と照らし合わせ、
最後まで揃っていれば自分で `.done` を書きます。自分で作っても問題ありません（[7](#7-done-ファイルの作り方)）。

`C:\StrataOffline` のコピーが不要なら、`robocopy` の代わりに `move` を使えばディスクを倍使わずに済みます。

### 6.3 画像エンコーダ（mmproj）

モデルのフォルダではなく、その **1つ上の `C:\Strata-data\models\`** に置き、**`.done` を必ず作ります**。

```bat
copy C:\StrataOffline\07_models\mmproj-Qwen3.8-Flash-Next-BF16.gguf C:\Strata-data\models\
type nul > C:\Strata-data\models\mmproj-Qwen3.8-Flash-Next-BF16.gguf.done
```

モデルの shard とは違い、setup は mmproj の中身を確かめて `.done` を作ることをしません。`.done` が無いと、
ファイルがあっても Hugging Face から落とし直そうとして止まります。

### 6.4 MTP ドラフト層

```bat
robocopy C:\StrataOffline\08_mtp\mtp C:\Strata-data\mtp /E
```

`C:\Strata-data\mtp\rt\experts.bin` があれば、setup は MTP の取得をしません。その代わりに SHA-256 の照合
（オフラインでできます）と、トークン表（`draft_vocab`）の配置だけを行います。

### 6.5 置いたあとの形

```
C:\Strata\
  .venv\               (pip.ini を含む)
  third_party\llama.cpp-3cf0325.zip
  third_party\llama.cpp-3cf0325.zip.done
  setup.py, START-HERE.bat, ...
C:\Strata-data\
  models\mmproj-Qwen3.8-Flash-Next-BF16.gguf
  models\mmproj-Qwen3.8-Flash-Next-BF16.gguf.done
  models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  mtp\rt\experts.bin   (ほか mtp 一式)
C:\StrataOffline\
  05_engine\strata-windows-x64.zip
  06_wheels\*.whl
```

---

## 7. `.done` ファイルの作り方

### 7.1 仕組み

setup.py は、ダウンロードを最後まで終えたファイルの隣に **`<元のファイル名>.done`** という目印ファイルを書きます。
次回からは、この目印があればそのファイルを「取得済み」とみなし、サーバーに問い合わせません
（`setup.py` の `done()` と `mark()`）。

- 判定は **ファイルがあるかどうかだけ** で、中身は見ません（setup 自身は日時を1行書きます）。空のファイルで構いません。
- 名前は **元のファイル名の全体 + `.done`** です。拡張子を置き換えるのではなく後ろに足します。
  例: `llama.cpp-3cf0325.zip` → `llama.cpp-3cf0325.zip.done`
- 置き場所は対象ファイルと **同じフォルダ** です。
- `.done` があると中身を確かめずに使うので、作る前に [3](#3-sha-256-を確かめる) で SHA-256 を確認しておいてください。
  やり直させたいときは `.done` を消します。

### 7.2 この手順で `.done` が必要なもの

| ファイル | `.done` | 理由 |
|---|---|---|
| `C:\Strata\third_party\llama.cpp-3cf0325.zip` | **必要** | 無いと GitHub から落とそうとして止まる |
| `C:\Strata-data\models\mmproj-Qwen3.8-Flash-Next-BF16.gguf` | **必要** | 無いと Hugging Face から落とそうとして止まる |
| モデルの shard（`…-0000N-of-00002.gguf`） | 任意 | setup が中身を確かめて自分で作る |
| エンジン zip | 不要 | `--prebuilt` のフォルダからコピーされる |
| MTP（`mtp\`） | 不要 | `rt\experts.bin` があるかどうかで判断される |

### 7.3 作り方（どれでも同じ結果になります）

**コマンドプロンプト**（空のファイル）

```bat
type nul > "C:\Strata\third_party\llama.cpp-3cf0325.zip.done"
```

**PowerShell**（空のファイル）

```powershell
New-Item -ItemType File -Path "C:\Strata\third_party\llama.cpp-3cf0325.zip.done"
```

**PowerShell**（setup.py と同じく日時を書く）

```powershell
Set-Content -Path "C:\Strata\third_party\llama.cpp-3cf0325.zip.done" -Value (Get-Date -Format "yyyy-MM-dd HH:mm") -Encoding utf8
```

**フォルダ内の全 GGUF にまとめて作る**（PowerShell。任意）

```powershell
Get-ChildItem C:\Strata-data\models\IQ3_S\*.gguf | ForEach-Object { New-Item -ItemType File -Path ($_.FullName + ".done") -Force }
```

**エクスプローラーで作る場合の注意**: 「新規作成 → テキスト文書」で作って名前を変えると、拡張子が非表示の設定では
`llama.cpp-3cf0325.zip.done.txt` になってしまいます。表示タブの「ファイル名拡張子」をオンにしてから名前を付けてください。
作ったら `dir C:\Strata\third_party` で名前を確かめます。

---

## 8. オフラインPC: setup を実行する

### 8.1 インストール（モデルの準備まで。起動はしない）

コマンドプロンプトで実行します。`--prebuilt` のフォルダ名の最後には `\` を付けます。

**タイプA（64 GB / RTX PRO 4000 Blackwell 24 GB）**

```bat
cd /d C:\Strata
START-HERE.bat --family qwen --model IQ2_XS --context 204800 --vision yes --yes --no-start --prebuilt C:\StrataOffline\05_engine\
```

**タイプB（256 GB / RTX 2000 Ada 16 GB）**

```bat
cd /d C:\Strata
START-HERE.bat --family qwen --model IQ3_S --context 204800 --vision yes --yes --no-start --prebuilt C:\StrataOffline\05_engine\
```

**タイプC（256 GB / 12 GB）**

```bat
cd /d C:\Strata
START-HERE.bat --family qwen --model IQ3_S --context 204800 --vision yes --yes --no-start --prebuilt C:\StrataOffline\05_engine\
```

オプションの意味:

| オプション | 意味 |
|---|---|
| `--family qwen` | Qwen3.8-Flash-Next 本体 |
| `--model` / `--context` | [1](#1-pc-ごとのモデルと設定) の表のとおり |
| `--vision yes` | 画像を使う。エンコーダは GPU で動く（`C:\Strata-data\models\` の mmproj と `.done` が必要） |
| `--yes` | 残りの質問は推奨の答えで進める（KV キャッシュは 8-bit、実験的機能はオフ） |
| `--no-start` | インストールだけして起動しない（次の `--calibrate` のため） |
| `--prebuilt <フォルダ>` | エンジン zip をこのフォルダからコピーする |

途中で出る主な表示と、その意味:

- `[ok] ... already installed`（Python パッケージ）、または pip がローカルの wheel からインストールする表示
- `Strata engine copied` と `could not get a SHA-256 for strata-windows-x64.zip from GitHub ... NOT verified.
  Installing it as it is.`: ローカルフォルダを指定したときの想定どおりの動作です（[3](#3-sha-256-を確かめる) で確認済みなら問題ありません）
- `llama.cpp source already downloaded`: `.done` が効いています
- `images: on`
- `KV streaming on: the context's KV cache lives in RAM (2.8 GB), more experts fit in VRAM`
  （判定に使うのは搭載 RAM の総量で、空き容量ではありません。3タイプとも条件を満たすので `KV streaming off: ...` は
  出ないはずです。出た場合は、Windows が認識している RAM の量を確かめてください。`--kv-streaming on` を足すと、
  判定に関係なく有効にできます）
- `model files present`
- `vision encoder already downloaded` と `vision encoder: C:\Strata-data\models\mmproj-Qwen3.8-Flash-Next-BF16.gguf`:
  mmproj の `.done` が効いています
- `MTP draft layer: C:\Strata-data\mtp\rt`
- 最後に `All set.`

**タイプC だけの表示**: VRAM が 14 GB 未満なので、「ドラフト層のトークン表を `--draft-vocab en` にすると VRAM が
空く」という提案が出ます。これは提案だけで、設定は変わりません。既定の `cjk` は日本語・中国語・韓国語を含むので、
**日本語で使うなら既定のままにしてください**。英語とコードしか使わない場合や、起動時に
"the draft head does not fit" と出て止まる場合だけ、`--draft-vocab en` を検討してください。

**タイプC だけの表示（画像）**: VRAM が 12.5 GB 以下で画像を GPU で処理する場合、setup は次の提案（tip）を出します。
「画像のリクエストが止まるようなら `--vram-reserve-mib 1000` で setup をやり直す」。これも提案だけで、設定は
変わりません。実際に画像のリクエストが止まったときだけ、8.1 のコマンドに `--vram-reserve-mib 1000` を足して
もう一度実行してください。足すと VRAM が少し多く空くので、文章の出力はわずかに遅くなります。

### 8.2 自動チューニング（PC ごとに必ず実行）

`--yes` や `--no-start` を付けると、setup はチューニングを勧めません。インストールのあとで別に実行します。

```bat
cd /d C:\Strata
START-HERE.bat --calibrate --no-start
```

- 実際のエンジンを起動し、内蔵の3つのプロンプトで出力速度を測ります。ネットには接続しません。約 5〜10 分かかり、その間 PC は重くなります。
- 測る設定は次の3つです。既定値より 3% 以上速くなったものだけを採用します。
  - `--pcie-frac`: GPU に無いエキスパートのうち、PCIe で GPU に送って計算する割合
  - `--spec-min-p`: ドラフトの推測をどこまで伸ばすか
  - `--pool-workers`: エキスパートを計算する CPU スレッドの数
- 結果は2か所に保存されます。
  - モデルの設定（`C:\Strata\strata-iq3_s.json` など）
  - PC ごとの記録（`%APPDATA%\Strata\settings.json`）
  
  同じ PC・同じモデル・同じコンテキストで入れ直すと、記録した値が自動で使われます。
- 記録は「GPU 名・VRAM・CPU・RAM・モデル・コンテキスト・画像の有無」ごとです。**他の PC で測った値は使えません**。
  同じタイプでも、PC ごとに実行してください。
- `the tuning did not finish` と出た場合は、既定の設定のまま使えます。エンジンのログの最後を確認してから、もう一度実行してください。

### 8.3 起動と確認

```bat
cd /d C:\Strata
START-HERE.bat
```

- ブラウザで <http://127.0.0.1:8080/> が開けば完了です。
- API の場所:
  - OpenAI 互換: `http://127.0.0.1:8080/v1`
  - Anthropic 互換: `http://127.0.0.1:8080/v1/messages`
- 2回目以降も `START-HERE.bat` だけで起動します。起動のたびに、入っているエンジンが setup.py の必要とする版
  （`MIN_ENGINE` = 0.1.40.2）以上かを確かめます。以上であれば、ネットには接続しません。
- 他の PC から使う場合は、必ず API キーを付けてください（付けないと LAN 内の誰でも使えます）。

  ```bat
  START-HERE.bat --host 0.0.0.0 --api-key <長いランダムな文字列>
  ```

---

## 9. 別のバージョンにするとき

モデルファイルと MTP ドラフト層はそのまま使い回せます。新しいタグの `setup.py` を開き、次の値を確かめて、
変わったものだけ集め直してください。

| setup.py の値 | 変わっていたら |
|---|---|
| `CMakeLists.txt` の `project(strata VERSION …)` と `MIN_ENGINE` | 同じタグの release からエンジン zip を落とす（ソースとエンジンのタグを必ず揃える） |
| `requirements.txt`、`CUDA_WHEELS` | wheel を落とし直す（[2.7](#27-python-パッケージwheel)） |
| `LLAMA_CPP_COMMIT` | その commit の zip を `llama.cpp-<先頭7文字>.zip` という名前で落とす |
| `HF_REVISIONS` | 新しいリビジョンのモデルを使う場合だけ落とし直す |

`UPDATE.bat` は `git pull` を使うので、オフラインPCでは使えません。新しいソースを別のフォルダ（例 `C:\Strata-0.1.41`）に
展開し、5〜8 と同じ手順で入れてください。データフォルダ（`C:\Strata-data`）は `%APPDATA%\Strata\settings.json` に
記録されているので、そのまま使われます。

---

## 付録A: うまくいかないとき

| 表示 | 原因と対処 |
|---|---|
| `cannot reach github.com` / `huggingface.co` | そのファイルの置き場所・名前・`.done` が違う。[6](#6-オフラインpc-ファイルを所定の場所に置く) と [7](#7-done-ファイルの作り方) を見直す |
| `could not find a version that satisfies the requirement …` | wheel が足りないか、Python のバージョンが違う（cp312 以外）。`.venv\Scripts\python.exe --version` を確認 |
| `the NVIDIA driver is too old` | ドライバを 580 以上にする |
| `not enough free disk space` | データフォルダのドライブの空きを増やす。別のドライブに置くなら `--data-dir D:\Strata-data` |
| `… is longer than its tensor` / shard の長さのエラー | モデルのコピーが壊れている。SHA-256 を確かめ、コピーし直して `.done` を消す |
| `some MTP tensors are not the checkpoint's` | MTP のコピーが壊れている（setup は取り直そうとして失敗する）。オンラインPCで `mtp_fetch.py verify` が通る `mtp\` を運び直す |
| 起動時に `the draft head does not fit` | VRAM 不足。`START-HERE.bat --draft-vocab en` で起動する（日本語の下書きは弱くなる） |
| `vision encoder` のところで `cannot reach huggingface.co` | mmproj が `C:\Strata-data\models\` に無いか、`.done` が無い（[6.3](#63-画像エンコーダmmproj)） |
| 画像のリクエストが止まる（12 GB の GPU） | 8.1 のコマンドに `--vram-reserve-mib 1000` を足して setup をやり直す |
