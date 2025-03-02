#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
フォン=ミーゼス分布のサンプル生成

このスクリプトは、フォン=ミーゼス分布に従ったサンプルを生成します。
"""

import numpy as np
from scipy import stats

def generate_von_mises_samples(mu=0.0, kappa=5.0, n_samples=100):
    """
    フォン=ミーゼス分布からサンプルを生成する関数
    
    Parameters:
    -----------
    mu : float
        平均方向（ラジアン単位、-π から π の範囲）
    kappa : float
        集中度パラメータ（大きいほど分布が集中する）
    n_samples : int
        生成するサンプル数
        
    Returns:
    --------
    samples : ndarray
        生成されたサンプル
    """
    # サンプル生成
    samples = stats.vonmises.rvs(kappa, loc=mu, size=n_samples)
    return samples

if __name__ == "__main__":
    # パラメータの設定
    mu = 0.0  # 平均方向（ラジアン）
    kappa = 5.0  # 集中度パラメータ
    n_samples = 100  # サンプル数
    
    # サンプル生成
    samples = generate_von_mises_samples(mu=mu, kappa=kappa, n_samples=n_samples)
    print(f"生成したサンプル（最初の10個）: {samples[:10]}")
    
    # パラメータを変更して再度サンプル生成
    print("\n異なるパラメータでのサンプル生成:")
    mu = np.pi/4  # 平均方向を変更
    kappa = 8.0   # 集中度を変更
    samples = generate_von_mises_samples(mu=mu, kappa=kappa, n_samples=n_samples)
    print(f"生成したサンプル（最初の10個）: {samples[:10]}") 