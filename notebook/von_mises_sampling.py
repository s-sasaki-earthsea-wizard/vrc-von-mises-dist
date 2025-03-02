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
    def display_polar_histogaram(samples, N=60):
        """
        サンプルの極座標ヒストグラムを表示

        Args:
            samples (numpy.ndarray): 表示するサンプル
        """
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
    def calculate_mean_concentration(samples):
        x = np.cos(samples)
        y = np.sin(samples)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        mean_direction = np.arctan2(mean_y, mean_x)
        mean_direction_deg = mean_direction * 180 / np.pi

        kappa, _, _ = stats.vonmises.fit(samples)
        return mean_direction_deg, kappa

