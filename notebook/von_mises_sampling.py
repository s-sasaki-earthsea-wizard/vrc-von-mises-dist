#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
フォン=ミーゼス分布のサンプル生成と可視化

このスクリプトは、フォン=ミーゼス分布に従ったサンプルを生成し、
ヒストグラムと円形プロットで可視化します。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# プロットの設定
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5)

def generate_von_mises_samples(mu=0.0, kappa=5.0, n_samples=100, show_plots=True):
    """
    フォン=ミーゼス分布からサンプルを生成し、可視化する関数
    
    Parameters:
    -----------
    mu : float
        平均方向（ラジアン単位、-π から π の範囲）
    kappa : float
        集中度パラメータ（大きいほど分布が集中する）
    n_samples : int
        生成するサンプル数
    show_plots : bool
        プロットを表示するかどうか
        
    Returns:
    --------
    samples : ndarray
        生成されたサンプル
    """
    # サンプル生成
    samples = stats.vonmises.rvs(kappa, loc=mu, size=n_samples)
    
    if show_plots:
        # ヒストグラムによる可視化
        plt.figure(figsize=(10, 6))
        plt.hist(samples, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 理論的な確率密度関数を重ねる
        x = np.linspace(-np.pi, np.pi, 1000)
        plt.plot(x, stats.vonmises.pdf(x, kappa, loc=mu), 'r-', lw=2, 
                 label=f'Von Mises PDF (μ={mu}, κ={kappa})')
        
        plt.title(f'フォン=ミーゼス分布のヒストグラム (μ={mu}, κ={kappa})')
        plt.xlabel('角度 (ラジアン)')
        plt.ylabel('確率密度')
        plt.xlim(-np.pi, np.pi)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 円形プロットによる可視化
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # サンプルを散布図としてプロット
        ax.scatter(samples, np.ones(n_samples), alpha=0.7, c='skyblue', edgecolor='black')
        
        # 理論的な確率密度関数を重ねる
        theta = np.linspace(-np.pi, np.pi, 1000)
        radii = stats.vonmises.pdf(theta, kappa, loc=mu)
        radii = radii / np.max(radii) * 0.9  # 正規化して見やすくする
        ax.plot(theta, radii, 'r-', lw=2, label=f'Von Mises PDF (μ={mu}, κ={kappa})')
        
        ax.set_title(f'フォン=ミーゼス分布の円形プロット (μ={mu}, κ={kappa})')
        ax.set_rticks([])  # 半径目盛りを非表示
        ax.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.show()
    
    return samples

def compare_kappa_values(mu=0.0, kappa_values=[0.5, 2.0, 5.0, 10.0]):
    """
    異なる集中度パラメータでのフォン=ミーゼス分布を比較する関数
    
    Parameters:
    -----------
    mu : float
        平均方向（ラジアン単位、-π から π の範囲）
    kappa_values : list
        比較する集中度パラメータのリスト
    """
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    plt.figure(figsize=(12, 8))
    x = np.linspace(-np.pi, np.pi, 1000)
    
    for i, k in enumerate(kappa_values):
        color = colors[i % len(colors)]
        plt.plot(x, stats.vonmises.pdf(x, k, loc=mu), color=color, lw=2, 
                 label=f'κ={k}')
    
    plt.title(f'異なる集中度パラメータでのフォン=ミーゼス分布 (μ={mu})')
    plt.xlabel('角度 (ラジアン)')
    plt.ylabel('確率密度')
    plt.xlim(-np.pi, np.pi)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_mu_values(mu_values=[-np.pi/2, 0, np.pi/4, np.pi/2], kappa=5.0):
    """
    異なる平均方向でのフォン=ミーゼス分布を比較する関数
    
    Parameters:
    -----------
    mu_values : list
        比較する平均方向のリスト
    kappa : float
        集中度パラメータ
    """
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    plt.figure(figsize=(12, 8))
    x = np.linspace(-np.pi, np.pi, 1000)
    
    for i, m in enumerate(mu_values):
        color = colors[i % len(colors)]
        plt.plot(x, stats.vonmises.pdf(x, kappa, loc=m), color=color, lw=2, 
                 label=f'μ={m:.2f}')
    
    plt.title(f'異なる平均方向でのフォン=ミーゼス分布 (κ={kappa})')
    plt.xlabel('角度 (ラジアン)')
    plt.ylabel('確率密度')
    plt.xlim(-np.pi, np.pi)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # パラメータの設定
    mu = 0.0  # 平均方向（ラジアン）
    kappa = 5.0  # 集中度パラメータ
    n_samples = 100  # サンプル数
    
    # サンプル生成と可視化
    samples = generate_von_mises_samples(mu=mu, kappa=kappa, n_samples=n_samples)
    print(f"生成したサンプル（最初の10個）: {samples[:10]}")
    
    # 異なる集中度パラメータでの比較
    compare_kappa_values(mu=mu)
    
    # 異なる平均方向での比較
    compare_mu_values(kappa=kappa)
    
    # パラメータを変更して再度サンプル生成
    print("\n異なるパラメータでのサンプル生成:")
    mu = np.pi/4  # 平均方向を変更
    kappa = 8.0   # 集中度を変更
    samples = generate_von_mises_samples(mu=mu, kappa=kappa, n_samples=n_samples)
    print(f"生成したサンプル（最初の10個）: {samples[:10]}")