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

class VonMisesDistAnalyzer:
    """フォン=ミーゼス分布のサンプル生成と分析を行うクラス
    
    このクラスは、フォン=ミーゼス分布からのサンプル生成、
    ヒストグラム表示、異なるパラメータでの分布比較などの
    機能を提供します。
    """
    @staticmethod
    def generate_von_mises_samples(mu=0.0, kappa=1.0, n_samples=10**4):
        """
        フォン=ミーゼス分布からサンプルを生成

        Args:
            mu (float, optional): 平均方向（ラジアン単位）. Defaults to 0.0.
            kappa (float, optional): 集中度パラメータ. Defaults to 1.0.
            n_samples (int, optional): 生成するサンプル数. Defaults to 1000.

        Returns:
            numpy.ndarray: 生成されたサンプル
        """    
        return stats.vonmises.rvs(kappa, loc=mu, size=n_samples)

    @staticmethod
    def display_histogram(samples):
        """
        サンプルのヒストグラムを表示

        Args:
            samples (numpy.ndarray): 表示するサンプル
        """
        samples = samples * 180/np.pi
        plt.figure(figsize=(6, 6))
        plt.hist(samples, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.xlabel('Angle')
        plt.ylabel('Sample number')
        plt.xlim(-180, 180)
        plt.xticks([-180, -90, 0, 90, 180], ['-180°', '-90°', '0°', '90°', '180°'])
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def display_polar_histogaram(samples):
        """
        サンプルの極座標ヒストグラムを表示

        Args:
            samples (numpy.ndarray): 表示するサンプル
        """
        N = 60
        bottom = 4
        max_height = 4
        
        # binとヒストグラムを計算
        bin_edges = np.linspace(-np.pi, np.pi, N + 1)
        hist, _ = np.histogram(samples, bins=bin_edges)

        # ビンの中心角度を計算
        theta = bin_edges[:-1] + np.diff(bin_edges) / 2
        
        # ヒストグラムの高さを正規化
        if hist.max() > 0:  # ゼロ除算を防ぐ
            hist = hist / hist.max() * max_height
        
        width = (2 * np.pi) / N  # バーの幅
        
        ax = plt.subplot(111, polar=True)
        bars = ax.bar(theta, hist, width=width, bottom=bottom)
        
        ax.set_rticks([])  # 半径目盛りを非表示
        ax.grid(True, alpha=0.3)

        for r, hist_bar in zip(hist, bars):
            hist_bar.set_facecolor(plt.cm.jet(r / max_height))
            hist_bar.set_alpha(0.8)

        plt.show()
        
    @staticmethod
    def compare_kappa_values(mu=0.0, kappa_values=[0.5, 2.0, 5.0, 10.0]):
        """
        異なる集中度パラメータでのフォン=ミーゼス分布を比較する関数
        
        Args:
            mu (float, optional): 平均方向（ラジアン単位、-π から π の範囲）. Defaults to 0.0.
            kappa_values (list, optional): 比較する集中度パラメータのリスト. Defaults to [0.5, 2.0, 5.0, 10.0].
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

    @staticmethod
    def compare_mu_values(mu_values=[-np.pi/2, 0, np.pi/4, np.pi/2], kappa=5.0):
        """
        異なる平均方向でのフォン=ミーゼス分布を比較する関数
        
        Args:
            mu_values (list, optional): 比較する平均方向のリスト. Defaults to [-np.pi/2, 0, np.pi/4, np.pi/2].
            kappa (float, optional): 集中度パラメータ. Defaults to 5.0.
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
    samples = VonMisesDistAnalyzer.generate_von_mises_samples(mu=mu, kappa=kappa, n_samples=n_samples)
    print(f"生成したサンプル（最初の10個）: {samples[:10]}")
    
    # 異なる集中度パラメータでの比較
    VonMisesDistAnalyzer.compare_kappa_values(mu=mu)
    
    # 異なる平均方向での比較
    VonMisesDistAnalyzer.compare_mu_values(kappa=kappa)
    
    # パラメータを変更して再度サンプル生成
    print("\n異なるパラメータでのサンプル生成:")
    mu = np.pi/4  # 平均方向を変更
    kappa = 8.0   # 集中度を変更
    samples = VonMisesDistAnalyzer.generate_von_mises_samples(mu=mu, kappa=kappa, n_samples=n_samples)
    print(f"生成したサンプル（最初の10個）: {samples[:10]}")