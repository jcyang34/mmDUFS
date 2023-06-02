# baseline methods to compare mvdufs with
import scipy
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
# compute distance
def compute_distance_np(X,Y=None):
    if Y is None:
        # return distance matrix
        #D = np.linalg.norm(X[:,:, np.newaxis] - X[:,:, np.newaxis].T, axis = 1)
        D = scipy.spatial.distance.cdist(X,X)
    else:
        #D = np.linalg.norm(X[:,:, np.newaxis] - Y[:,:, np.newaxis].T, axis = 1)
        D = scipy.spatial.distance.cdist(X,Y)
    return D


# compute graph laplacian given the data matrix, numpy implementation
def full_affinity_knn_np(X, knn=2,fac=0.6, laplacian="normalized"):

    D = compute_distance_np(X)
    # compute sigma 
    D_copy = D.copy()
    D_copy = D + np.diag(np.repeat(float("inf"), D.shape[0]))
    nn = D_copy.argsort(axis = 1)

    kth_idx = nn[:,knn-1]
    kth_dist = D[range(D.shape[0]), kth_idx]

    sigma = np.median(kth_dist) if np.median(kth_dist) >= 1e-8 else 1e-8 
    W = np.exp(- D ** 2 / (fac * sigma))
    Dsum= np.sum(W,axis=1)

    if laplacian == "normalized":
        Dminus_half = np.diag(np.power(Dsum,-0.5))
        P = Dminus_half@W@Dminus_half
    elif laplacian == "random_walk":
        Dminus=np.diag(np.power(Dsum,-1))
        P=Dminus@W
    elif laplacian == "unnormalized":
        P = W
    return P,Dsum,W

# compute laplacian scores of the features from each modality on the kernel
def LS_graph(X,Gx):
    score_vec = np.diagonal(X.T@Gx@X).reshape(-1)
    score_vec  = score_vec / np.linalg.norm(X,axis=0)
    
    return score_vec


# compute laplacian scores of the features from each modality on the joint kernel
def LS_joint_graph(X,Y,P_joint):
    score_vec_X = np.diagonal(X.T@P_joint@X).reshape(-1)
    score_vec_X  = score_vec_X / np.linalg.norm(X,axis=0)

    score_vec_Y = np.diagonal(Y.T@P_joint@Y).reshape(-1)
    score_vec_Y = score_vec_Y / np.linalg.norm(Y,axis=0)

    return score_vec_X,score_vec_Y

# given the scores of the feature vector, rank the features and return an indicator vector with 1 being the top k features
def top_n_feats(score_vec,n):
    top_n_vec = np.zeros(len(score_vec))
    top_n_vec[np.argsort(score_vec)[::-1][:n]] = 1
    return top_n_vec

def fs_eval(X,Y,label_true_X,label_true_Y,baselines = ["concat","sum","prod"],nx=50,ny=50, knn=2,fac=0.6, laplacian="normalized"):
    eval_dict = {}
    if "concat" in baselines:
        _,_,top_k_X,top_k_Y = concatenation_fs(X,Y,nx,ny, knn,fac, laplacian)
        eval_dict["concat"] = {"X":f1_score(label_true_X,top_k_X),"Y": f1_score(label_true_Y,top_k_Y)}
    if "sum" in baselines:
        _,_,top_k_X,top_k_Y = kernel_sum_fs(X,Y,nx,ny, knn,fac, laplacian)
        eval_dict["sum"] = {"X":f1_score(label_true_X,top_k_X),"Y": f1_score(label_true_Y,top_k_Y)}
    if "prod" in baselines:
        _,_,top_k_X,top_k_Y = kernel_prod_fs(X,Y,nx,ny, knn,fac, laplacian)
        eval_dict["prod"] = {"X":f1_score(label_true_X,top_k_X),"Y": f1_score(label_true_Y,top_k_Y)}

    return eval_dict

# baseline 1: concatenate two matrices, then construct a graph, the compute the scores
def concatenation_fs(X,Y,nx=50,ny=50, knn=2,fac=0.6, laplacian="normalized"):
    
    # conatenate X Y
    XY = np.concatenate((X,Y),axis=1)

    # construct a graph based on the concatenated mat
    G_xy,_,_ = full_affinity_knn_np(XY,knn=knn,fac=fac,laplacian=laplacian)
    
    # return the scores of the features, along with the top k feature indicators
    score_vec_X,score_vec_Y = LS_joint_graph(X,Y,G_xy)
    top_k_X = top_n_feats(score_vec_X,n=nx)
    top_k_Y = top_n_feats(score_vec_Y,n=ny)
    return score_vec_X,score_vec_Y,top_k_X,top_k_Y


    
# baseline 2: construct the kernel separately, then compute the kernel sum as the joint kernel, then compute the scores
def kernel_sum_fs(X,Y, nx=50,ny=50, knn=2,fac=0.6, laplacian="normalized"):

    # construct the kernel separately
    G_x,_,_ = full_affinity_knn_np(X,knn=knn,fac=fac,laplacian=laplacian)
    G_y,_,_ = full_affinity_knn_np(Y,knn=knn,fac=fac,laplacian=laplacian)

    # kernel sum
    G_xy = G_x + G_y

    # return the scores of the features, along with the top k feature indicators
    score_vec_X,score_vec_Y = LS_joint_graph(X,Y,G_xy)
    top_k_X = top_n_feats(score_vec_X,n=nx)
    top_k_Y = top_n_feats(score_vec_Y,n=ny)
   
    return score_vec_X,score_vec_Y,top_k_X,top_k_Y

# baseline 3: construct the kernel separately, then compute the kernel product as the joint kernel, then compute the scores
def kernel_prod_fs(X,Y,nx=50, ny = 50, knn=2,fac=0.6, laplacian="normalized"):

    # construct the kernel separately
    G_x,_,_ = full_affinity_knn_np(X,knn=knn,fac=fac,laplacian=laplacian)
    G_y,_,_ = full_affinity_knn_np(Y,knn=knn,fac=fac,laplacian=laplacian)

    # kernel sum
    G_xy = G_x@G_y

    # return the scores of the features, along with the top k feature indicators
    score_vec_X,score_vec_Y = LS_joint_graph(X,Y,G_xy)
    top_k_X = top_n_feats(score_vec_X,n=nx)
    top_k_Y = top_n_feats(score_vec_Y,n=ny)
   
    return score_vec_X,score_vec_Y,top_k_X,top_k_Y


def kernel_prod_fs_v2(X,Y,nx=50, ny = 50, knn=2,fac=0.6, laplacian="normalized"):

    # construct the kernel separately
    G_x,_,_ = full_affinity_knn_np(X,knn=knn,fac=fac,laplacian=laplacian)
    G_y,_,_ = full_affinity_knn_np(Y,knn=knn,fac=fac,laplacian=laplacian)

    # kernel sum
    G_xy = G_x@G_y + G_y@G_x

    # return the scores of the features, along with the top k feature indicators
    score_vec_X,score_vec_Y = LS_joint_graph(X,Y,G_xy)
    top_k_X = top_n_feats(score_vec_X,n=nx)
    top_k_Y = top_n_feats(score_vec_Y,n=ny)
   
    return score_vec_X,score_vec_Y,top_k_X,top_k_Y



def fs_eval_diff(X,Y,label_true_X,label_true_Y,baselines = ["concat","sum","prod"],n_total_x = 100,n_total_y = 100, 
                 nx=50,ny=50, knn=2,fac=0.6, laplacian="normalized"):
    
    # get the smooth features of X and Y
    Gx,_,_ = full_affinity_knn_np(X,knn=knn,fac=fac,laplacian=laplacian)
    Gy,_,_ = full_affinity_knn_np(Y,knn=knn,fac=fac,laplacian=laplacian)
    top_feats_x = top_n_feats(LS_graph(X,Gx),n_total_x)
    top_feats_y = top_n_feats(LS_graph(Y,Gy),n_total_y)
    
    # get the shared features from each baselines
    eval_dict = {}
    diff_list = {}
    if "concat" in baselines:
        _,_,top_k_X,top_k_Y = concatenation_fs(X,Y,nx,ny, knn,fac, laplacian)
        
        diff_X = 1*((top_feats_x - top_k_X)>0)
        diff_Y = 1*((top_feats_y - top_k_Y)>0)
        
        eval_dict["concat"] = {"X":f1_score(label_true_X,diff_X),"Y": f1_score(label_true_Y,diff_Y)}
        diff_list["concat"] = {"X":diff_X,"Y":diff_Y}
    if "sum" in baselines:
        _,_,top_k_X,top_k_Y = kernel_sum_fs(X,Y,nx,ny, knn,fac, laplacian)
        
        diff_X = 1*((top_feats_x - top_k_X)>0)
        diff_Y = 1*((top_feats_y - top_k_Y)>0)
        
        eval_dict["sum"] = {"X":f1_score(label_true_X,diff_X),"Y": f1_score(label_true_Y,diff_Y)}
        diff_list["sum"] = {"X":diff_X,"Y":diff_Y}
    if "prod" in baselines:
        _,_,top_k_X,top_k_Y = kernel_prod_fs(X,Y,nx,ny, knn,fac, laplacian)
        diff_X = 1*((top_feats_x - top_k_X)>0)
        diff_Y = 1*((top_feats_y - top_k_Y)>0)
        eval_dict["prod"] = {"X":f1_score(label_true_X,diff_X),"Y": f1_score(label_true_Y,diff_Y)}
        diff_list["prod"] = {"X":diff_X,"Y":diff_Y}
    return eval_dict, diff_list



