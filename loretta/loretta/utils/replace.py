import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def cosine_similarity(v1, v2):
    """
    compute cosine similarity between v1 and v2
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 
    return dot_product / (norm_v1 * norm_v2)

def kmeans_cosine(X, k, max_iter=100, tol=1e-4):
    """
    K-means clustering using cosine similarity as distance metric
    X: input data
    k: cluster number
    max_iter: mAaximum number of iterations
    tol: threshold to stop the iteration
    
    return labels and centers
    """
    # 初始化聚类中心（随机选择k个样本作为初始聚类中心）
    # initialize cluster centers (randomly choose k samples as initial cluster centers)
    n_samples, n_features = X.shape
    centers = X[np.random.choice(n_samples, k, replace=False)]
    
    
    for _ in range(max_iter):
        # 步骤1: 为每个样本分配最近的聚类中心（基于余弦相似度）
        # step 1: assign each sample to the nearest cluster center (based on cosine similarity)
        new_labels = np.array([np.argmax([cosine_similarity(x, c) for c in centers]) for x in X])
        
        # 步骤2: 计算新的聚类中心
        # step 2: calculate new cluster centers
        new_centers = np.zeros_like(centers)
        for i in range(k):
            cluster_points = X[new_labels == i]
            if len(cluster_points) > 0:
                # 对于每个簇，计算所有点的平均向量作为新的质心
                # for each cluster, calculate the average vector of all points as the new centroid
                new_centers[i] = np.mean(cluster_points, axis=0)
        
        # 步骤3: 检查是否收敛（如果聚类中心变化小于阈值）
        # step 3: check if converged (if the change of cluster centers is less than the threshold)
        if np.all(np.abs(new_centers - centers) < tol):
            break
        
        # 更新聚类中心
        # update cluster centers
        centers = new_centers
    
    return new_labels, centers

# def empty_clustering(X, k):
#     n_samples = X.shape[0]  
#     length = n_samples // k
#     last_length = n_samples - length * (k - 1)
    
#     centers = np.array([X[i * length] for i in range(k)])
#     labels = np.concatenate([np.full(length, i) for i in range(k - 1)])
#     labels = np.concatenate((labels, np.full(last_length, k - 1)))
    
#     return labels,centers

def find_best_k(A, B):

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    if A.ndim == 1:
        A = A.reshape(-1, 1)  

    if B.ndim == 1:
        B = B.reshape(-1, 1)  

    A_T_A = np.dot(A.T, A)  
    A_T_B = np.dot(A.T, B) 

    k = np.dot(np.linalg.inv(A_T_A), A_T_B)  
    return k

def decompose_weight(weight: torch.Tensor,vector_length):
    m,n=weight.shape
    new_weight=np.empty((m,n))
 
    for split in range(n/vector_length):
        data_subset = weight[split*vector_length:(split+1)*vector_length,:]
        X=data_subset.transpose(0, 1).numpy()
        labels, centers=kmeans_cosine(X,20)
    
        for i in range(m):
            A=centers[labels[i]]
            B=X[i]      
            k=np.dot(A,B)/np.dot(A,A)
            new_weight[i,split*vector_length:(split+1)*vector_length]=k*centers[labels[i]]

    return torch.from_numpy(new_weight.transpose(0, 1))
