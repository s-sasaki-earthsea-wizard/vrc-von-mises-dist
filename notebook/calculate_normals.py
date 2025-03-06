#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
円と正方形の点群とその法線ベクトルを計算するモジュール

このモジュールは、円と正方形の点群を生成し、
各点における法線ベクトルを計算する関数を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from mpl_toolkits.mplot3d import Axes3D


class NormalStatsCalculator:
    """法線ベクトルの統計解析を行うクラス
    
    このクラスは、円と正方形の点群を生成し、
    法線ベクトルの計算、統計解析、および可視化の機能を提供します。
    """
    
    @staticmethod
    def generate_circle_points(N, radius=1.0, noise=0.0):
        """
        円周上の点群を生成する関数

        Args:
            N (int): 生成する点の数
            radius (float, optional): 円の半径. Defaults to 1.0.
            noise (float, optional): 点に加えるノイズの大きさ. Defaults to 0.0.

        Returns:
            tuple: (points, normals) - 点の座標と法線ベクトルのNumPy配列
        """
        # 円周上に均等に点を配置
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        
        # 点の座標を計算
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points = np.column_stack((x, y))
        
        # ノイズを加える（オプション）
        if noise > 0:
            noise_vector = np.random.normal(0, noise, size=points.shape)
            points += noise_vector
        
        # 法線ベクトルを計算（中心から外向き）
        normals = points / np.sqrt(np.sum(points**2, axis=1))[:, np.newaxis]
        
        return points, normals

    @staticmethod
    def generate_square_points(N, side_length=1.0, noise=0.0):
        """
        正方形の周囲に点群を生成する関数

        Args:
            N (int): 生成する点の数（各辺にN//4個の点を配置）
            side_length (float, optional): 正方形の一辺の長さ. Defaults to 1.0.
            noise (float, optional): 点に加えるノイズの大きさ. Defaults to 0.0.

        Returns:
            tuple: (points, normals) - 点の座標と法線ベクトルのNumPy配列
        """
        # 各辺に配置する点の数を計算
        points_per_side = N // 4
        remaining_points = N % 4
        
        # 正方形の半分のサイズ
        half_size = side_length / 2
        
        # 各辺に沿って点を生成
        points = []
        normals = []
        
        # 上辺
        top_points = points_per_side + (1 if remaining_points > 0 else 0)
        x_top = np.linspace(-half_size, half_size, top_points)
        y_top = np.ones(top_points) * half_size
        points.extend(np.column_stack((x_top, y_top)))
        normals.extend(np.column_stack((np.zeros(top_points), np.ones(top_points))))
        
        # 右辺
        right_points = points_per_side + (1 if remaining_points > 1 else 0)
        x_right = np.ones(right_points) * half_size
        y_right = np.linspace(half_size, -half_size, right_points)[1:]  # 重複を避ける
        if len(y_right) > 0:  # 空の配列を避ける
            points.extend(np.column_stack((x_right[:-1], y_right)))
            normals.extend(np.column_stack((np.ones(len(y_right)), np.zeros(len(y_right)))))
        
        # 下辺
        bottom_points = points_per_side + (1 if remaining_points > 2 else 0)
        x_bottom = np.linspace(half_size, -half_size, bottom_points)[1:]  # 重複を避ける
        y_bottom = np.ones(len(x_bottom)) * -half_size
        if len(x_bottom) > 0:  # 空の配列を避ける
            points.extend(np.column_stack((x_bottom, y_bottom)))
            normals.extend(np.column_stack((np.zeros(len(x_bottom)), -np.ones(len(x_bottom)))))
        
        # 左辺
        left_points = points_per_side
        x_left = np.ones(left_points) * -half_size
        y_left = np.linspace(-half_size, half_size, left_points)[1:-1]  # 重複を避ける
        if len(y_left) > 0:  # 空の配列を避ける
            points.extend(np.column_stack((x_left[:len(y_left)], y_left)))
            normals.extend(np.column_stack((-np.ones(len(y_left)), np.zeros(len(y_left)))))
        
        # NumPy配列に変換
        points = np.array(points)
        normals = np.array(normals)
        
        # ノイズを加える（オプション）
        if noise > 0:
            noise_vector = np.random.normal(0, noise, size=points.shape)
            points += noise_vector
        
        return points, normals

    @staticmethod
    def calculate_normal_statistics(normals):
        """
        法線ベクトルの統計情報を計算する関数

        Args:
            normals (numpy.ndarray): 法線ベクトルの配列

        Returns:
            dict: 法線ベクトルの統計情報
        """
        # 法線ベクトルの平均
        mean_normal = np.mean(normals, axis=0)
        
        # 法線ベクトルの角度を計算
        angles = np.arctan2(normals[:, 1], normals[:, 0])
        
        # 角度の平均と集中度を計算（フォン=ミーゼス分布のパラメータ推定）
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        R = np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
        kappa = R * (2 - R**2) / (1 - R**2) if R < 0.53 else -0.4 + 1.39 * R + 0.43 / (1 - R)
        
        return {
            "mean_normal": mean_normal,
            "mean_angle": mean_angle,
            "concentration": kappa,
            "R": R
        }

    @staticmethod
    def plot_points_and_normals(points, normals, title="Points and Normals"):
        """
        点群と法線ベクトルを可視化する関数

        Args:
            points (numpy.ndarray): 点の座標の配列
            normals (numpy.ndarray): 法線ベクトルの配列
            title (str, optional): プロットのタイトル. Defaults to "Points and Normals".
        """
        plt.figure(figsize=(10, 10))
        
        # 点をプロット
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=30, label='Points')
        
        # 法線ベクトルを矢印でプロット
        for i in range(len(points)):
            plt.arrow(points[i, 0], points[i, 1], 
                    normals[i, 0] * 0.2, normals[i, 1] * 0.2,
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        # 平均法線ベクトルを計算して表示
        stats = NormalStatsCalculator.calculate_normal_statistics(normals)
        mean_normal = stats["mean_normal"]
        plt.arrow(0, 0, mean_normal[0] * 0.5, mean_normal[1] * 0.5,
                head_width=0.1, head_length=0.1, fc='green', ec='green', 
                label=f'Mean Normal: ({mean_normal[0]:.3f}, {mean_normal[1]:.3f})')
        
        # プロットの設定
        plt.axis('equal')
        plt.grid(True)
        plt.title(f"{title}\nMean Angle: {stats['mean_angle']:.3f}, Concentration: {stats['concentration']:.3f}")
        plt.legend()
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        
        plt.show()

    @staticmethod
    def plot_normal_distribution(normals, title="Normal Vector Distribution"):
        """
        法線ベクトルの分布を極座標でプロットする関数

        Args:
            normals (numpy.ndarray): 法線ベクトルの配列
            title (str, optional): プロットのタイトル. Defaults to "Normal Vector Distribution".
        """
        # 法線ベクトルの角度を計算
        angles = np.arctan2(normals[:, 1], normals[:, 0])
        
        # 統計情報を計算
        stats = NormalStatsCalculator.calculate_normal_statistics(normals)
        
        # 極座標プロット
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # ヒストグラム
        bins = np.linspace(-np.pi, np.pi, 37)  # 10度ごとに区切る
        hist, bin_edges = np.histogram(angles, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # ヒストグラムをプロット
        ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], alpha=0.5)
        
        # フォン=ミーゼス分布をプロット（集中度が十分ある場合）
        if stats["concentration"] > 0.1:
            x = np.linspace(-np.pi, np.pi, 1000)
            y = scipy_stats.vonmises.pdf(x, stats["concentration"], loc=stats["mean_angle"])
            ax.plot(x, y, 'r-', lw=2, label=f'Von Mises PDF (μ={stats["mean_angle"]:.2f}, κ={stats["concentration"]:.2f})')
        
        ax.set_title(title)
        ax.set_rticks([])  # 半径目盛りを非表示
        ax.grid(True)
        
        plt.show()


def main():
    """
    メイン関数：円と正方形の点群と法線ベクトルを生成して可視化
    """
    N = 100  # 点の数
    
    # 円の点群と法線ベクトルを生成
    circle_points, circle_normals = NormalStatsCalculator.generate_circle_points(N)
    circle_stats = NormalStatsCalculator.calculate_normal_statistics(circle_normals)
    
    print("円の法線ベクトルの統計情報:")
    print(f"平均法線ベクトル: {circle_stats['mean_normal']}")
    print(f"平均角度: {circle_stats['mean_angle']}")
    print(f"集中度: {circle_stats['concentration']}")
    print(f"R値: {circle_stats['R']}")
    
    # 正方形の点群と法線ベクトルを生成
    square_points, square_normals = NormalStatsCalculator.generate_square_points(N)
    square_stats = NormalStatsCalculator.calculate_normal_statistics(square_normals)
    
    print("\n正方形の法線ベクトルの統計情報:")
    print(f"平均法線ベクトル: {square_stats['mean_normal']}")
    print(f"平均角度: {square_stats['mean_angle']}")
    print(f"集中度: {square_stats['concentration']}")
    print(f"R値: {square_stats['R']}")
    
    # 可視化
    NormalStatsCalculator.plot_points_and_normals(circle_points, circle_normals, "Circle Points and Normals")
    NormalStatsCalculator.plot_points_and_normals(square_points, square_normals, "Square Points and Normals")
    
    # 法線ベクトルの分布を可視化
    NormalStatsCalculator.plot_normal_distribution(circle_normals, "Circle Normal Vector Distribution")
    NormalStatsCalculator.plot_normal_distribution(square_normals, "Square Normal Vector Distribution")


if __name__ == "__main__":
    main()
