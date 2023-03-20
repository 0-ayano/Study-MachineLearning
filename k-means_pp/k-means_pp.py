import numpy as np
import matplotlib.pyplot as plt

class KMeans_pp:
    """
    名前 : __init__
    引数 : 設定するクラスタ数(n_clusters)、ループの上限(max_iter)
    説明 : 適当なデータセットを作成する
    """
    def __init__(self, n_clusters, max_iter = 1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    """
    名前 : Create_Data
    引数 : 生成するデータ数(qty)
    説明 : 適当なデータセットを作成する
    """
    def Create_Data(self, qty):
        np.random.seed(0)
        points1 = np.random.randn(qty, 2)
        points2 = np.random.randn(qty, 2) + np.array([4,0])
        points3 = np.random.randn(qty, 2) + np.array([5,8])

        points = np.r_[points1, points2, points3]
        np.random.shuffle(points)
        return points

    """
    名前 : main
    引数 : データ(Data)
    説明 : k-means++
    """
    def Kmeans_pp(self, Data):
        # ランダムで最初のクラスタ点を決定
        random_point = np.random.choice(np.array(range(Data.shape[0])))
        first_cluster = Data[random_point]
        first_cluster = first_cluster[np.newaxis,:]

        # 最初のクラスタ点とそれ以外のデータ点との距離の2乗を計算し、それぞれをその総和で割る
        p = ((Data - first_cluster)**2).sum(axis = 1) / ((Data - first_cluster)**2).sum()
        r =  np.random.choice(np.array(range(Data.shape[0])), size = 1, replace = False, p = p)
        first_cluster = np.r_[first_cluster ,Data[r]]

        #分割するクラスター数が3個以上の場合
        if self.n_clusters >= 3:
            #指定の数のクラスタ点を指定できるまで繰り返し
            while first_cluster.shape[0] < self.n_clusters:
                #各クラスター点と各データポイントとの距離の2乗を算出
                dist_f = ((Data[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :])**2).sum(axis = 1)
                #最も距離の近いクラスター点はどれか導出
                f_argmin = dist_f.argmin(axis = 1)
                #最も距離の近いクラスター点と各データポイントとの距離の2乗を導出
                for i in range(dist_f.shape[1]):
                    dist_f.T[i][f_argmin != i] = 0

                #新しいクラスタ点を確率的に導出
                pp = dist_f.sum(axis = 1) / dist_f.sum()
                rr = np.random.choice(np.array(range(Data.shape[0])), size = 1, replace = False, p = pp)
                #新しいクラスター点を初期値として加える
                first_cluster = np.r_[first_cluster ,Data[rr]]

        #最初のラベルづけを行う
        dist = (((Data[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :]) ** 2).sum(axis = 1))
        self.labels_ = dist.argmin(axis = 1)
        labels_prev = np.zeros(Data.shape[0])
        count = 0
        self.cluster_centers_ = np.zeros((self.n_clusters, Data.shape[1]))

        #各データポイントが属しているクラスターが変化しなくなった、又は一定回数の繰り返しを越した場合は終了
        while (not (self.labels_ == labels_prev).all() and count < self.max_iter):
            #その時点での各クラスターの重心を計算する
            for i in range(self.n_clusters):
                same_label = Data[self.labels_ == i, :]
                self.cluster_centers_[i, :] = same_label.mean(axis = 0)
            #各データポイントと各クラスターの重心間の距離を総当たりで計算する
            dist = ((Data[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2).sum(axis = 1)
            #1つ前のクラスターラベルを覚えておく。1つ前のラベルとラベルが変化しなければプログラムは終了する。
            labels_prev = self.labels_
            #再計算した結果、最も距離の近いクラスターのラベルを割り振る
            self.labels_ = dist.argmin(axis = 1)
            count += 1
            self.count = count   


model =  KMeans_pp(3)
Data  = model.Create_Data(100)
model.Kmeans_pp(Data)

print(model.labels_)

markers = ["+", "*", "o", '+']
color = ['r', 'b', 'g', 'k']
for i in range(4):
    p = Data[model.labels_ == i, :]
    plt.scatter(p[:, 0], p[:, 1], marker = markers[i], color = color[i])

plt.show()