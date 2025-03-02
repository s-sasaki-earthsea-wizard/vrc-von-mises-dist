# フォン=ミーゼス分布のサンプル生成と可視化

このリポジトリには、フォン=ミーゼス分布に従ったサンプルを生成し、可視化するためのコードが含まれています。

## フォン=ミーゼス分布とは

フォン=ミーゼス分布は、円周上の確率分布で、方向データの解析によく使われます。この分布は2つのパラメータを持ちます：

- `mu`: 平均方向（ラジアン単位、-π から π の範囲）
- `kappa`: 集中度パラメータ（大きいほど分布が集中する）

## ファイル構成

- `requirements.txt`: 必要なPythonパッケージのリスト
- `von_mises_sampling.py`: フォン=ミーゼス分布のサンプル生成と可視化のための関数を含むPythonスクリプト
- `simple_von_mises.py`: フォン=ミーゼス分布のサンプル生成のみを行うシンプルなPythonスクリプト

## 環境設定

必要なパッケージをインストールするには、以下のコマンドを実行してください：

```bash
pip install -r requirements.txt
```

## 使用方法

### シンプルなサンプル生成

`simple_von_mises.py`を実行すると、フォン=ミーゼス分布に従った100個のサンプルが生成されます：

```bash
python simple_von_mises.py
```

### サンプル生成と可視化

`von_mises_sampling.py`を実行すると、サンプル生成と可視化が行われます：

```bash
python von_mises_sampling.py
```

### パラメータのカスタマイズ

スクリプト内の以下の変数を変更することで、サンプル生成のパラメータをカスタマイズできます：

```python
mu = 0.0  # 平均方向（ラジアン）
kappa = 5.0  # 集中度パラメータ
n_samples = 100  # サンプル数
```

## 関数の使用例

```python
import numpy as np
from von_mises_sampling import generate_von_mises_samples

# パラメータの設定
mu = np.pi/4  # 平均方向
kappa = 8.0   # 集中度
n_samples = 100  # サンプル数

# サンプル生成と可視化
samples = generate_von_mises_samples(mu=mu, kappa=kappa, n_samples=n_samples)
```

## 参考文献

- [SciPy - Von Mises Distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html)
- [Wikipedia - Von Mises Distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution) 