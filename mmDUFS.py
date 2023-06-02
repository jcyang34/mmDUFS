import os
import numpy as np

import scipy
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

from sklearn.cluster import SpectralClustering
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from munkres import Munkres
import sklearn.metrics as ms
from sklearn.metrics import f1_score
from collections import Counter

class DataSet:
    """Base data set class
    """

    def __init__(self, shuffle=True, labeled=False, **data_dict):
        assert '_data1' in data_dict
        assert '_data2' in data_dict
        assert data_dict['_data1'].shape[0] == data_dict['_data2'].shape[0]
        if labeled:
            assert '_labels' in data_dict
            assert data_dict['_data1'].shape[0] == data_dict['_labels'].shape[0]
            assert data_dict['_data2'].shape[0] == data_dict['_labels'].shape[0]
        self._labeled = labeled
        self._shuffle = shuffle
        self.__dict__.update(data_dict)
        self._num_samples = self._data1.shape[0]
        self._index_in_epoch = 0
        if self._shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self._data1)

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def data1(self):
        return self._data1
    
    @property
    def data2(self):
        return self._data2

    @property
    def labels(self):
        return self._labels

    @property
    def labeled(self):
        return self._labeled

    @property
    def valid_data1(self):
        return self._valid_data1

    @property
    def valid_data2(self):
        return self._valid_data2

    @property
    def valid_labels(self):
        return self._valid_labels

    @property
    def test_data1(self):
        return self._test_data1
    
    @property
    def test_data2(self):
        return self._test_data2


    @property
    def test_labels(self):
        return self._test_labels

    @classmethod
    def load(cls, filename):
        data_dict = np.load(filename)
        return cls(**data_dict)

    def save(self, filename):
        data_dict = self.__dict__
        np.savez_compressed(filename, **data_dict)

    def _shuffle_data(self):
        shuffled_idx = np.arange(self._num_samples)
        np.random.shuffle(shuffled_idx)
        self._data1 = self._data1[shuffled_idx]
        self._data2 = self._data2[shuffled_idx]
        if self._labeled:
            self._labels = self._labels[shuffled_idx]
    
    def get_amuont_batchs(self,batch_size):
        return int(np.ceil(self._num_samples/batch_size))
    
    def next_batch(self, batch_size):
        assert batch_size <= self._num_samples
        start = self._index_in_epoch
        if start + batch_size > self._num_samples:
            data_batch1 = self._data1[start:]
            data_batch2 = self._data2[start:]
            if self._labeled:
                labels_batch = self._labels[start:]
            remaining = batch_size - (self._num_samples - start)
            if self._shuffle:
                self._shuffle_data()
            start = 0
            data_batch1 = np.concatenate([data_batch1, self._data1[:remaining]],
                                        axis=0)
            data_batch2 = np.concatenate([data_batch2, self._data2[:remaining]],
                                        axis=0)
            if self._labeled:
                labels_batch = np.concatenate([labels_batch,
                                               self._labels[:remaining]],
                                              axis=0)
            self._index_in_epoch = remaining
        else:
            data_batch1 = self._data1[start:start + batch_size]
            data_batch2 = self._data2[start:start + batch_size]
            if self._labeled:
                labels_batch = self._labels[start:start + batch_size]
            self._index_in_epoch = start + batch_size
        batch = (data_batch1, data_batch2, labels_batch) if self._labeled else (data_batch1,data_batch2)
        return batch

    
class DataSet3:
    """Base data set class
    """

    def __init__(self, shuffle=True, labeled=True, **data_dict):
        assert '_data1' in data_dict
        assert '_data2' in data_dict
        assert '_data3' in data_dict
        assert data_dict['_data1'].shape[0] == data_dict['_data2'].shape[0]  == data_dict['_data3'].shape[0]
        if labeled:
            assert '_labels' in data_dict
            assert data_dict['_data1'].shape[0] == data_dict['_labels'].shape[0]
            assert data_dict['_data2'].shape[0] == data_dict['_labels'].shape[0]
            assert data_dict['_data3'].shape[0] == data_dict['_labels'].shape[0]
        self._labeled = labeled
        self._shuffle = shuffle
        self.__dict__.update(data_dict)
        self._num_samples = self._data1.shape[0]
        self._index_in_epoch = 0
        if self._shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self._data1)

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def data1(self):
        return self._data1
    
    @property
    def data2(self):
        return self._data2
    @property
    def data3(self):
        return self._data3

    @property
    def labels(self):
        return self._labels

    @property
    def labeled(self):
        return self._labeled

    @property
    def valid_data1(self):
        return self._valid_data1

    @property
    def valid_data2(self):
        return self._valid_data2

    @property
    def valid_data3(self):
        return self._valid_data3
    
    @property
    def valid_labels(self):
        return self._valid_labels

    @property
    def test_data1(self):
        return self._test_data1
    
    @property
    def test_data2(self):
        return self._test_data2
    
    @property
    def test_data3(self):
        return self._test_data3

    @property
    def test_labels(self):
        return self._test_labels

    @classmethod
    def load(cls, filename):
        data_dict = np.load(filename)
        return cls(**data_dict)

    def save(self, filename):
        data_dict = self.__dict__
        np.savez_compressed(filename, **data_dict)

    def _shuffle_data(self):
        shuffled_idx = np.arange(self._num_samples)
        np.random.shuffle(shuffled_idx)
        self._data1 = self._data1[shuffled_idx]
        self._data2 = self._data2[shuffled_idx]
        self._data3 = self._data3[shuffled_idx]
        if self._labeled:
            self._labels = self._labels[shuffled_idx]
    
    def get_amuont_batchs(self,batch_size):
        return int(np.ceil(self._num_samples/batch_size))
    
    def next_batch(self, batch_size):
        assert batch_size <= self._num_samples
        start = self._index_in_epoch
        if start + batch_size > self._num_samples:
            data_batch1 = self._data1[start:]
            data_batch2 = self._data2[start:]
            data_batch3 = self._data3[start:]
            if self._labeled:
                labels_batch = self._labels[start:]
            remaining = batch_size - (self._num_samples - start)
            if self._shuffle:
                self._shuffle_data()
            start = 0
            data_batch1 = np.concatenate([data_batch1, self._data1[:remaining]],
                                        axis=0)
            data_batch2 = np.concatenate([data_batch2, self._data2[:remaining]],
                                        axis=0)
            data_batch3 = np.concatenate([data_batch3, self._data3[:remaining]],
                                        axis=0)
            if self._labeled:
                labels_batch = np.concatenate([labels_batch,
                                               self._labels[:remaining]],
                                              axis=0)
            self._index_in_epoch = remaining
        else:
            data_batch1 = self._data1[start:start + batch_size]
            data_batch2 = self._data2[start:start + batch_size]
            data_batch3 = self._data3[start:start + batch_size]
            if self._labeled:
                labels_batch = self._labels[start:start + batch_size]
            self._index_in_epoch = start + batch_size
        batch = (data_batch1, data_batch2, data_batch3, labels_batch) if self._labeled else (data_batch1,data_batch2,data_batch3)
        return batch


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def squared_distance(X):
    '''
    Calculates the squared Euclidean distance matrix.

    X:              an n-by-p matrix, which includes n samples in dimension p

    returns:        n x n pairwise squared Euclidean distance matrix
    '''

    r = tf.reduce_sum(X*X, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(X, X, transpose_b=True) + tf.transpose(r)
    return D


def full_affinity_knn(X, knn=2,fac=0.6, laplacian="normalized"):
    '''
    Calculates the symmetrized full Gaussian affinity matrix, the used kernel width is the median over the 
    k-nearst neighbor of all the given points in the given dataset times an input scale factor.

    X:              an n-by-p matrix, which includes n samples in dimension p
    knn:            the k in the k-nearest neighbor that will be used in order to determin the kernel width
    fac:            the scale factor of the 
    laplacian:      "normalized", "random_walk", or "unnormalized"
    returns:        n x n affinity matrix
    '''

    Dx = squared_distance(X)    
    nn = tf.nn.top_k(-Dx, knn, sorted=True)
    knn_distances = -nn[0][:, knn - 1]
    mu=tf.contrib.distributions.percentile(knn_distances,50.,interpolation='higher')
    ml = tf.contrib.distributions.percentile(knn_distances,50.,interpolation='lower')
    sigma=(mu+ml)/2.
    sigma=tf.cond(tf.less(sigma,1e-8),lambda:1.,lambda:sigma)
    W = K.exp(-Dx/ (fac*sigma) )
    Dsum=K.sum(W,axis=1)
    
    if laplacian == "normalized":
        Dminus_half = tf.linalg.diag(K.pow(Dsum,-0.5))
        P = tf.matmul(tf.matmul(Dminus_half,W),Dminus_half)
    elif laplacian == "random_walk":
        Dminus=tf.linalg.diag(K.pow(Dsum,-1))
        P=tf.matmul(Dminus,W)
    elif laplacian == "unnormalized":
        P = W
    else:
        raise KeyError("laplacian parameter not supported: {}".format(laplacian))
    return P,Dsum,W





# Shared operator

class JointModel(object): 
     def __init__(self,
            input_dim1,
            input_dim2,
                  batch_size,
            seed=1,
            lam1=0.1,
            lam2=0.1,
            fac=5,
            knn=2,
            laplacian = "normalized",
                  const = 1e-3,
                  const2 = 1,
            is_param_free_loss=False
        ):
        # Register hyperparameters for feature selection
        self.fac = fac
        self.knn=knn
        self.laplacian = laplacian
        self.sigma = 0.5
        self.lam1 = lam1
        self.lam2 = lam2
        self.input_dim1=input_dim1
        self.input_dim2=input_dim2
        self.batch_size = batch_size
        self.is_param_free_loss= is_param_free_loss
        self.const = const
        self.const2 = const2

        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            # tf Graph Input
            X1 = tf.placeholder(tf.float32, [None, input_dim1]) # i.e. mnist data image of shape 28*28=784            
            X2 = tf.placeholder(tf.float32, [None, input_dim2])


            self.learning_rate= tf.placeholder(tf.float32, (), name='learning_rate')
            
            self.nnweights = []
            # first modality
            masked_input1 = X1
            with tf.variable_scope('concrete1', reuse=tf.AUTO_REUSE):
                self.alpha1 = tf.get_variable('alpha1', [input_dim1,],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                masked_input1 = self.feature_selector1(masked_input1)
            
            # second modality
            masked_input2 = X2
            with tf.variable_scope('concrete2', reuse=tf.AUTO_REUSE):
                self.alpha2 = tf.get_variable('alpha2', [input_dim2,],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                masked_input2 = self.feature_selector2(masked_input2)

            # compute kernel for each modality    
            Pn1,D1,W1=full_affinity_knn(masked_input1,knn=self.knn,fac=self.fac,laplacian = self.laplacian)            
            Pn2,D2,W2=full_affinity_knn(masked_input2,knn=self.knn,fac=self.fac,laplacian = self.laplacian)            
            
            ## gates regularization
            input2cdf1 = self.alpha1
            reg1 = 0.5 - 0.5*tf.erf((-1/2 - input2cdf1)/(self.sigma*np.sqrt(2)))
            reg_gates1 = tf.reduce_mean(reg1)+tf.constant(1e-6)
            
            input2cdf2 = self.alpha2
            reg2 = 0.5 - 0.5*tf.erf((-1/2 - input2cdf2)/(self.sigma*np.sqrt(2)))
            reg_gates2 = tf.reduce_mean(reg2)+tf.constant(1e-6)
            

            # construct the joint operator
            # current implementation: Pn is the random walk laplacian, P_joint is symmetric
            
            P_joint = tf.matmul(Pn2,tf.transpose(Pn1)) + tf.matmul(Pn1,tf.transpose(Pn2))
            # const2: a const to balance the scores compared to the gates
            P_joint = P_joint * self.const2

            
            # compute the scores for each modality on the joint op
            laplacian_score1 = -tf.reduce_mean(tf.matmul(P_joint,masked_input1)*masked_input1)
            laplacian_score2 = -tf.reduce_mean(tf.matmul(P_joint,masked_input2)*masked_input2)
            
            # compute the loss
            loss = laplacian_score1 + laplacian_score2 +  reg_gates1*self.lam1 + reg_gates2*self.lam2
            

            self.reg_gates1 = reg_gates1 # for debugging
            self.reg_gates2 = reg_gates2

            # Gradient Descent
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            

            accuracy=tf.Variable([0],tf.float32)
            
            # Initialize the variables (i.e. assign their default value)
            init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()
            
        # Save into class members
        self.X1 = X1
        self.X2 = X2

        self.W1 = W1
        self.W2 = W2

        self.loss = loss
        self.laplacian_score1 = laplacian_score1
        self.laplacian_score2 = laplacian_score2
        self.kern1=Pn1
        self.kern2=Pn2
        self.ker_joint = P_joint
        self.train_step = train_step
        # set random state
        tf.set_random_seed(seed)
        # Initialize all global variables
        self.sess.run(init_op)
        
     def _to_tensor(self, x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)

     def hard_sigmoid(self, x):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -0.5`, `1.` if `x > 0.5`.
        In `-0.5 <= x <= 0.5`, returns `x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = x + 0.5
        zero = _to_tensor(0., x.dtype.base_dtype)
        one = _to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, one)
        return x
    
     # feature selector for the first modality
     def feature_selector1(self, x):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param x - input. shape==[batch_size, feature_num]
        :return: gated input
        '''
        base_noise = tf.random_normal(shape=tf.shape(x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha1, axis=0) + self.sigma * base_noise 
        stochastic_gate = self.hard_sigmoid(z)
        masked_x = x * stochastic_gate
        return masked_x

     # feature selector for the second modality
     def feature_selector2(self, x):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param x - input. shape==[batch_size, feature_num]
        :return: gated input
        '''
        base_noise = tf.random_normal(shape=tf.shape(x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha2, axis=0) + self.sigma * base_noise 
        stochastic_gate = self.hard_sigmoid(z)
        masked_x = x * stochastic_gate
        return masked_x
    

    
    
     def get_raw_alpha1(self):
        """
        evaluate the learned dropout rate
        """
        dp_alpha = self.sess.run(self.alpha1)
        return dp_alpha

     def get_prob_alpha1(self):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha1 = self.get_raw_alpha1()
        prob_gate = self.compute_learned_prob(dp_alpha1)
        return prob_gate
    
    
     def get_raw_alpha2(self):
        """
        evaluate the learned dropout rate
        """
        dp_alpha = self.sess.run(self.alpha2)
        return dp_alpha

     def get_prob_alpha2(self):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha2 = self.get_raw_alpha2()
        prob_gate = self.compute_learned_prob(dp_alpha2)
        return prob_gate


    
    
     def hard_sigmoid_np(self, x):
        return np.minimum(1, np.maximum(0,x+0.5))
    
     def compute_learned_prob(self, alpha):
        return self.hard_sigmoid_np(alpha)

     def load(self, model_path=None):
        if model_path == None:
            raise Exception()
        self.saver.restore(self.sess, model_path)

     def save(self, step, model_dir=None):
        if model_dir == None:
            raise Exception()
        try:
            os.mkdir(model_dir)
        except:
            pass
        model_file = model_dir + "/model"
        self.saver.save(self.sess, model_file, global_step=step)
 
     def train(self, dataset, learning_rate,feature_label1=None,feature_label2=None,display_step=100, num_epoch=100, labeled=False):
        losses = []
        LS1 = [] # ls for modality 1
        LS2 = [] # ls for modality 2
        reg_arr1 =[] 
        reg_arr2 = []
        precision_arr1=[]
        precision_arr2=[]
        recall_arr1=[]
        recall_arr2=[]
        f1_score_list1 = []
        f1_score_list2 = []
        Spectral_Kmeans_acc_arr=[]
        self.display_step=display_step
        print("num_samples : {}".format(dataset.num_samples))
        for epoch in range(num_epoch):
            avg_loss = 0. 
            avg_score1=0.
            avg_score2=0.  
            reg_loss1=0.
            reg_loss2=0.

            # Loop over all batches
            amount_batchs= dataset.get_amuont_batchs(self.batch_size)
            for i in range(amount_batchs):
                if labeled:
                    batch_xs1,batch_xs2 ,batch_ys= dataset.next_batch(self.batch_size)
                else:
                     batch_xs1,batch_xs2 = dataset.next_batch(self.batch_size)
                _, loss, laplacian_score1, laplacian_score2, reg_fs1,reg_fs2,ker_joint,kern1,kern2 = self.sess.run([self.train_step, self.loss, self.laplacian_score1,self.laplacian_score2,self.reg_gates1, self.reg_gates2,self.ker_joint,self.kern1,self.kern2], \
                                                          feed_dict={self.X1: batch_xs1, self.X2: batch_xs2, self.learning_rate:learning_rate})
                
                avg_loss += loss / amount_batchs
                avg_score1 += laplacian_score1 / amount_batchs
                avg_score2 += laplacian_score2 / amount_batchs
                reg_loss1 += reg_fs1 / amount_batchs
                reg_loss2 += reg_fs2 / amount_batchs
            
            alpha_p1 = self.get_prob_alpha1()
            alpha_p2 = self.get_prob_alpha2()
            precision1=np.sum(alpha_p1[:2])/np.sum(alpha_p1[:])
            precision2=np.sum(alpha_p2[:2])/np.sum(alpha_p2[:])
            recall1=np.sum(alpha_p1[:2])/2
            recall2=np.sum(alpha_p2[:2])/2
            
            losses.append(avg_loss)
            LS1.append(avg_score1)
            LS2.append(avg_score2)
            reg_arr1.append(reg_loss1)
            reg_arr2.append(reg_loss2)
            recall_arr1.append(recall1)
            recall_arr2.append(recall2)
            precision_arr1.append(precision1)
            precision_arr2.append(precision2)
            
            # compute the F1 score between the ground truth features and the selected features
            f1_s1=0
            f1_s2=0
            if feature_label1 is not None:
                f1_s1 = f1_score(feature_label1,alpha_p1>0.5)
                f1_s2 = f1_score(feature_label2,alpha_p2>0.5)
            f1_score_list1.append(f1_s1)
            f1_score_list2.append(f1_s2)
            
             
            if (epoch+1) % self.display_step == 0:
                if feature_label1 is not None:
                    print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "score1=", "{:.9f}".format(avg_score1)\
                      , "score2=", "{:.9f}".format(avg_score2), "reg1=", "{:.9f}".format(reg_fs1), "reg2=", "{:.9f}".format(reg_fs2)\
                      , "f1 - Mod1 = ","{:.4f}".format(f1_s1),"f1 - Mod2 = ","{:.4f}".format(f1_s2))

                else:
                    print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "score1=", "{:.9f}".format(avg_score1)\
                      , "score2=", "{:.9f}".format(avg_score2), "reg1=", "{:.9f}".format(reg_fs1), "reg2=", "{:.9f}".format(reg_fs2))
  
        print("Optimization Finished!")
        
        # compute the ls for each modality
        self.myker_joint = ker_joint
        self.mykern1 = kern1
        self.mykern2 = kern2
        #self.myy = batch_ys
        self.myX1 = batch_xs1
        self.myX2 = batch_xs2

        LS_list1 = []
        for i in range(self.myX1.shape[1]):
            LS_list1.append((self.myX1[:,i].reshape(1,-1)@self.myker_joint@self.myX1[:,i].reshape(-1,1))[0][0]/(np.linalg.norm(self.myX1[:,i])**2))
        
        LS_list2 = []
        for i in range(self.myX2.shape[1]):
            LS_list2.append((self.myX2[:,i].reshape(1,-1)@self.myker_joint@self.myX2[:,i].reshape(-1,1))[0][0]/(np.linalg.norm(self.myX2[:,i])**2))

        return LS1,LS2,losses,reg_arr1,reg_arr2,precision_arr1,precision_arr2,recall_arr1,recall_arr2,f1_score_list1,f1_score_list2,LS_list1,LS_list2



# Difference Operator
class DiffModel(object): 
     def __init__(self,
            input_dim1,
            input_dim2,
                  batch_size,
            seed=1,
            lam1=0.1,
            fac=5,
            knn=2,
            laplacian="normalized",
                  const = 1e-3,
                  const2 = 1e-3,
                  is_param_free_loss=False
        ):
        # Register hyperparameters for feature selection
        self.fac = fac
        self.knn=knn
        self.laplacian = laplacian
        self.sigma = 0.5
        self.lam1 = lam1
        self.input_dim1=input_dim1
        self.input_dim2=input_dim2
        self.batch_size=batch_size
        self.is_param_free_loss= is_param_free_loss
        self.const = const
        self.const2 = const2

        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            # tf Graph Input
            X1 = tf.placeholder(tf.float32, [None, input_dim1]) # i.e. mnist data image of shape 28*28=784            
            X2 = tf.placeholder(tf.float32, [None, input_dim2])


            self.learning_rate= tf.placeholder(tf.float32, (), name='learning_rate')
            
            self.nnweights = []
            # first modality, with feature selection
            masked_input1 = X1
            with tf.variable_scope('concrete1', reuse=tf.AUTO_REUSE):
                self.alpha1 = tf.get_variable('alpha1', [input_dim1,],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                masked_input1 = self.feature_selector1(masked_input1)

            # second modality, without feature selection
            
            # compute kernel for each modality    
            Pn1,D1,W1=full_affinity_knn(masked_input1,knn=self.knn,fac=self.fac,laplacian = self.laplacian)            
            Pn2,D2,W2=full_affinity_knn(X2,knn=self.knn,fac=self.fac,laplacian = self.laplacian)            
            
            ## gates regularization 
            input2cdf1 = self.alpha1
            reg1 = 0.5 - 0.5*tf.erf((-1/2 - input2cdf1)/(self.sigma*np.sqrt(2)))
            reg_gates1 = tf.reduce_mean(reg1)+tf.constant(1e-6)
            

            # construct the difference operator

            P2_inv = tf.linalg.inv(Pn2+self.const*tf.eye(self.batch_size)) # Bug: tf.identity
            P_diff = tf.matmul(tf.matmul(P2_inv,Pn1),P2_inv)
            P_diff = P_diff*(self.const2)
            
            # legact codes for normalized operator

            #D1_min_half = tf.linalg.diag(K.pow(D1,-0.5))
            #D2_min_half = tf.linalg.diag(K.pow(D2,-0.5))
            #L1 = tf.matmul(tf.matmul(D1_min_half,W1),D1_min_half)
            #L2 = tf.matmul(tf.matmul(D2_min_half,W2),D2_min_half)
            #P_tmp1 = tf.linalg.inv(L2+self.const*tf.identity(L2))
            #P_joint1 = tf.matmul(tf.matmul(P_tmp1,L1),P_tmp1)
            #P_joint1 = P_joint1*(const2)
            

            #P_tmp2 = tf.linalg.inv(W1+self.const*tf.identity(W1))
            #P_joint2 = tf.matmul(tf.matmul(P_tmp2,W2),P_tmp2)
            #P_joint2 = P_joint2*(const2)
                
            laplacian_score1 = -tf.reduce_mean(tf.matmul(P_diff,masked_input1)*masked_input1)
            
            loss = laplacian_score1 + reg_gates1*self.lam1
 
            self.reg_gates1 = reg_gates1 # for debugging

            # Gradient Descent
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            

            accuracy=tf.Variable([0],tf.float32)
            # Initialize the variables (i.e. assign their default value)
            init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()
            
        # Save into class members
        self.X1 = X1
        self.X2 = X2

        self.W1 = W1
        self.W2 = W2

        self.loss = loss
        self.laplacian_score1 = laplacian_score1
        self.kern1=Pn1
        self.kern2=Pn2
        self.kern2_inv = P2_inv
        self.ker_diff = P_diff
        self.train_step = train_step
        # set random state
        tf.set_random_seed(seed)
        # Initialize all global variables
        self.sess.run(init_op)
        
     def _to_tensor(self, x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)

     def hard_sigmoid(self, x):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -0.5`, `1.` if `x > 0.5`.
        In `-0.5 <= x <= 0.5`, returns `x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = x + 0.5
        zero = _to_tensor(0., x.dtype.base_dtype)
        one = _to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, one)
        return x
    
     # feature selector for the first modality
     def feature_selector1(self, x):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param x - input. shape==[batch_size, feature_num]
        :return: gated input
        '''
        base_noise = tf.random_normal(shape=tf.shape(x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha1, axis=0) + self.sigma * base_noise 
        stochastic_gate = self.hard_sigmoid(z)
        masked_x = x * stochastic_gate
        return masked_x    

    
    
     def get_raw_alpha1(self):
        """
        evaluate the learned dropout rate
        """
        dp_alpha = self.sess.run(self.alpha1)
        return dp_alpha

     def get_prob_alpha1(self):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha1 = self.get_raw_alpha1()
        prob_gate = self.compute_learned_prob(dp_alpha1)
        return prob_gate
        
     def hard_sigmoid_np(self, x):
        return np.minimum(1, np.maximum(0,x+0.5))
    
     def compute_learned_prob(self, alpha):
        return self.hard_sigmoid_np(alpha)

     def load(self, model_path=None):
        if model_path == None:
            raise Exception()
        self.saver.restore(self.sess, model_path)

     def save(self, step, model_dir=None):
        if model_dir == None:
            raise Exception()
        try:
            os.mkdir(model_dir)
        except:
            pass
        model_file = model_dir + "/model"
        self.saver.save(self.sess, model_file, global_step=step)
 
     def train(self, dataset, learning_rate,feature_label=None,display_step=100, num_epoch=100, labeled=False):
        losses = []
        LS1 = [] # ls for modality 1
        LS2 = [] # ls for modality 2
        reg_arr1 =[] 
        reg_arr2 = []
        precision_arr1=[]
        precision_arr2=[]
        recall_arr1=[]
        recall_arr2=[]
        f1_score_list = []
        Spectral_Kmeans_acc_arr=[]
        self.display_step=display_step
        
        print("num_samples : {}".format(dataset.num_samples))
        for epoch in range(num_epoch):
            avg_loss = 0. 
            avg_score1=0.
            avg_score2=0.  
            reg_loss1=0.
            reg_loss2=0.

            # Loop over all batches
            amount_batchs= dataset.get_amuont_batchs(self.batch_size)
            for i in range(amount_batchs):
                if labeled:
                    batch_xs1,batch_xs2 ,batch_ys= dataset.next_batch(self.batch_size)
                else:
                     batch_xs1,batch_xs2 = dataset.next_batch(self.batch_size)
                _, loss, laplacian_score1, reg_fs1,ker_diff,kern1,kern2,kern2_inv = self.sess.run([self.train_step, self.loss, 
                                                                  self.laplacian_score1,
                                                                  self.reg_gates1,self.ker_diff,self.kern1,self.kern2,self.kern2_inv], \
                                                          feed_dict={self.X1: batch_xs1, self.X2: batch_xs2, self.learning_rate:learning_rate})
                
                avg_loss += loss / amount_batchs
                avg_score1 += laplacian_score1 / amount_batchs
                reg_loss1 += reg_fs1 / amount_batchs
            
            alpha_p1 = self.get_prob_alpha1()
            precision1=np.sum(alpha_p1[:2])/np.sum(alpha_p1[:])
            recall1=np.sum(alpha_p1[:2])/2
            
            losses.append(avg_loss)
            LS1.append(avg_score1)
            reg_arr1.append(reg_loss1)
            recall_arr1.append(recall1)
            precision_arr1.append(precision1)
            
            # compute the F1 score between the ground truth features and the selected features
            f1_s = 0
            if feature_label is not None:
                f1_s = f1_score(feature_label,alpha_p1>0.5)
            f1_score_list.append(f1_s)
            
            
            #if epoch == 0:
            if (epoch+1) % self.display_step == 0:
                if feature_label is not None:
                    print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "score1=", "{:.9f}".format(avg_score1)\
                      , "reg1=", "{:.9f}".format(reg_fs1), "f1=", "{:.4f}".format(f1_s))
                else:
                    print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "score1=", "{:.9f}".format(avg_score1)\
                      , "reg1=", "{:.9f}".format(reg_fs1))

        print("Optimization Finished!")

        self.myker_diff = ker_diff
        self.mykern1 = kern1
        self.mykern2 = kern2
        self.mykern2_inv = kern2_inv
        #self.myy = batch_ys
        self.myX1 = batch_xs1
        self.myX2 = batch_xs2

        # compute the LS for each feature
        LS_list1 = []
        for i in range(self.myX1.shape[1]):
            LS_list1.append((self.myX1[:,i].reshape(1,-1)@self.myker_diff@self.myX1[:,i].reshape(-1,1))[0][0]/(np.linalg.norm(self.myX1[:,i])**2))
        
        return LS1,losses,reg_arr1,precision_arr1,recall_arr1,f1_score_list,LS_list1


class JointModel3(object): 
     def __init__(self,
            input_dim1,
            input_dim2,
                  input_dim3,
                  batch_size,
            seed=1,
            lam1=0.1,
            lam2=0.1,
                  lam3=0.1,
            fac=5,
            knn=2,
            laplacian = "normalized",
                  const = 1e-3,
                  const2 = 1,
            is_param_free_loss=False
        ):
        # Register hyperparameters for feature selection
        self.fac = fac
        self.knn=knn
        self.laplacian = laplacian
        self.sigma = 0.5
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.input_dim1=input_dim1
        self.input_dim2=input_dim2
        self.input_dim3 = input_dim3
        self.batch_size = batch_size
        self.is_param_free_loss= is_param_free_loss
        self.const = const
        self.const2 = const2

        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            # tf Graph Input
            X1 = tf.placeholder(tf.float32, [None, input_dim1]) # i.e. mnist data image of shape 28*28=784            
            X2 = tf.placeholder(tf.float32, [None, input_dim2])
            X3 = tf.placeholder(tf.float32, [None, input_dim3])

            self.learning_rate= tf.placeholder(tf.float32, (), name='learning_rate')
            
            self.nnweights = []
            # first modality
            masked_input1 = X1
            with tf.variable_scope('concrete1', reuse=tf.AUTO_REUSE):
                self.alpha1 = tf.get_variable('alpha1', [input_dim1,],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                masked_input1 = self.feature_selector1(masked_input1)
            
            # second modality
            masked_input2 = X2
            with tf.variable_scope('concrete2', reuse=tf.AUTO_REUSE):
                self.alpha2 = tf.get_variable('alpha2', [input_dim2,],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                masked_input2 = self.feature_selector2(masked_input2)
            
            # third modality
            masked_input3 = X3
            with tf.variable_scope('concrete3', reuse=tf.AUTO_REUSE):
                self.alpha3 = tf.get_variable('alpha3', [input_dim3,],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                masked_input3 = self.feature_selector3(masked_input3)

            # compute kernel for each modality    
            Pn1,D1,W1=full_affinity_knn(masked_input1,knn=self.knn,fac=self.fac,laplacian = self.laplacian)            
            Pn2,D2,W2=full_affinity_knn(masked_input2,knn=self.knn,fac=self.fac,laplacian = self.laplacian)   
            Pn3,D3,W3=full_affinity_knn(masked_input3,knn=self.knn,fac=self.fac,laplacian = self.laplacian)
            
            ## gates regularization
            input2cdf1 = self.alpha1
            reg1 = 0.5 - 0.5*tf.erf((-1/2 - input2cdf1)/(self.sigma*np.sqrt(2)))
            reg_gates1 = tf.reduce_mean(reg1)+tf.constant(1e-6)
            
            input2cdf2 = self.alpha2
            reg2 = 0.5 - 0.5*tf.erf((-1/2 - input2cdf2)/(self.sigma*np.sqrt(2)))
            reg_gates2 = tf.reduce_mean(reg2)+tf.constant(1e-6)
            
            input2cdf3 = self.alpha3
            reg3 = 0.5 - 0.5*tf.erf((-1/2 - input2cdf3)/(self.sigma*np.sqrt(2)))
            reg_gates3 = tf.reduce_mean(reg3)+tf.constant(1e-6)
            

            # construct the joint operator
            # current implementation: Pn is the random walk laplacian, P_joint is symmetric
            
            P_joint = tf.matmul(Pn2,tf.transpose(Pn1)) + tf.matmul(Pn1,tf.transpose(Pn2)) \
                    + tf.matmul(Pn3,tf.transpose(Pn1)) + tf.matmul(Pn1,tf.transpose(Pn3)) \
                    + tf.matmul(Pn2,tf.transpose(Pn3)) + tf.matmul(Pn3,tf.transpose(Pn2)) 
            # const2: a const to balance the scores compared to the gates
            P_joint = P_joint * self.const2

            
            # compute the scores for each modality on the joint op
            laplacian_score1 = -tf.reduce_mean(tf.matmul(P_joint,masked_input1)*masked_input1)
            laplacian_score2 = -tf.reduce_mean(tf.matmul(P_joint,masked_input2)*masked_input2)
            laplacian_score3 = -tf.reduce_mean(tf.matmul(P_joint,masked_input3)*masked_input3)
            # compute the loss
            loss = laplacian_score1 + laplacian_score2 + laplacian_score3 + reg_gates1*self.lam1 + reg_gates2*self.lam2  + reg_gates3*self.lam3
            

            self.reg_gates1 = reg_gates1 # for debugging
            self.reg_gates2 = reg_gates2
            self.reg_gates3 = reg_gates3
            # Gradient Descent
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            

            accuracy=tf.Variable([0],tf.float32)
            
            # Initialize the variables (i.e. assign their default value)
            init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()
            
        # Save into class members
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        
        self.loss = loss
        self.laplacian_score1 = laplacian_score1
        self.laplacian_score2 = laplacian_score2
        self.laplacian_score3 = laplacian_score3
        
        self.kern1=Pn1
        self.kern2=Pn2
        self.kern3=Pn3
        
        self.ker_joint = P_joint
        self.train_step = train_step
        # set random state
        tf.set_random_seed(seed)
        # Initialize all global variables
        self.sess.run(init_op)
        
     def _to_tensor(self, x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)

     def hard_sigmoid(self, x):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -0.5`, `1.` if `x > 0.5`.
        In `-0.5 <= x <= 0.5`, returns `x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = x + 0.5
        zero = _to_tensor(0., x.dtype.base_dtype)
        one = _to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, one)
        return x
    
     # feature selector for the first modality
     def feature_selector1(self, x):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param x - input. shape==[batch_size, feature_num]
        :return: gated input
        '''
        base_noise = tf.random_normal(shape=tf.shape(x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha1, axis=0) + self.sigma * base_noise 
        stochastic_gate = self.hard_sigmoid(z)
        masked_x = x * stochastic_gate
        return masked_x

     # feature selector for the second modality
     def feature_selector2(self, x):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param x - input. shape==[batch_size, feature_num]
        :return: gated input
        '''
        base_noise = tf.random_normal(shape=tf.shape(x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha2, axis=0) + self.sigma * base_noise 
        stochastic_gate = self.hard_sigmoid(z)
        masked_x = x * stochastic_gate
        return masked_x
    
    # feature selector for the second modality
     def feature_selector3(self, x):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param x - input. shape==[batch_size, feature_num]
        :return: gated input
        '''
        base_noise = tf.random_normal(shape=tf.shape(x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha3, axis=0) + self.sigma * base_noise 
        stochastic_gate = self.hard_sigmoid(z)
        masked_x = x * stochastic_gate
        return masked_x

    
    
     def get_raw_alpha1(self):
        """
        evaluate the learned dropout rate
        """
        dp_alpha = self.sess.run(self.alpha1)
        return dp_alpha

     def get_prob_alpha1(self):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha1 = self.get_raw_alpha1()
        prob_gate = self.compute_learned_prob(dp_alpha1)
        return prob_gate
    
    
     def get_raw_alpha2(self):
        """
        evaluate the learned dropout rate
        """
        dp_alpha = self.sess.run(self.alpha2)
        return dp_alpha

     def get_prob_alpha2(self):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha2 = self.get_raw_alpha2()
        prob_gate = self.compute_learned_prob(dp_alpha2)
        return prob_gate

     def get_raw_alpha3(self):
        """
        evaluate the learned dropout rate
        """
        dp_alpha = self.sess.run(self.alpha3)
        return dp_alpha

     def get_prob_alpha3(self):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha3 = self.get_raw_alpha3()
        prob_gate = self.compute_learned_prob(dp_alpha3)
        return prob_gate
    
    
     def hard_sigmoid_np(self, x):
        return np.minimum(1, np.maximum(0,x+0.5))
    
     def compute_learned_prob(self, alpha):
        return self.hard_sigmoid_np(alpha)

     def load(self, model_path=None):
        if model_path == None:
            raise Exception()
        self.saver.restore(self.sess, model_path)

     def save(self, step, model_dir=None):
        if model_dir == None:
            raise Exception()
        try:
            os.mkdir(model_dir)
        except:
            pass
        model_file = model_dir + "/model"
        self.saver.save(self.sess, model_file, global_step=step)
 
     def train(self, dataset, learning_rate,feature_label1=None,feature_label2=None,feature_label3=None,display_step=100, num_epoch=100, labeled=False):
        losses = []
        LS1 = [] # ls for modality 1
        LS2 = [] # ls for modality 2
        LS3 = [] # ls for modality 3
        reg_arr1 =[] 
        reg_arr2 = []
        reg_arr3 = []
        f1_score_list1 = []
        f1_score_list2 = []
        f1_score_list3 = []
        Spectral_Kmeans_acc_arr=[]
        self.display_step=display_step
        print("num_samples : {}".format(dataset.num_samples))
        for epoch in range(num_epoch):
            avg_loss = 0. 
            avg_score1=0.
            avg_score2=0.  
            avg_score3=0.
            reg_loss1=0.
            reg_loss2=0.
            reg_loss3=0.
            # Loop over all batches
            amount_batchs= dataset.get_amuont_batchs(self.batch_size)
            for i in range(amount_batchs):
                if labeled:
                    batch_xs1,batch_xs2 ,batch_xs3 ,batch_ys= dataset.next_batch(self.batch_size)
                else:
                     batch_xs1,batch_xs2,batch_xs3 = dataset.next_batch(self.batch_size)
                _, loss, laplacian_score1, laplacian_score2,laplacian_score3, reg_fs1,reg_fs2,reg_fs3,ker_joint,kern1,kern2,kern3 = self.sess.run([self.train_step,self.loss,self.laplacian_score1,self.laplacian_score2,self.laplacian_score3,self.reg_gates1,self.reg_gates2,self.reg_gates3,self.ker_joint,self.kern1,self.kern2,self.kern3],feed_dict={self.X1: batch_xs1,self.X2: batch_xs2,self.X3:batch_xs3,self.learning_rate:learning_rate})
                avg_loss += loss / amount_batchs
                avg_score1 += laplacian_score1 / amount_batchs
                avg_score2 += laplacian_score2 / amount_batchs
                avg_score3 += laplacian_score3 / amount_batchs
                
                reg_loss1 += reg_fs1 / amount_batchs
                reg_loss2 += reg_fs2 / amount_batchs
                reg_loss3 += reg_fs3 / amount_batchs
            alpha_p1 = self.get_prob_alpha1()
            alpha_p2 = self.get_prob_alpha2()
            alpha_p3 = self.get_prob_alpha3()
            
            
            losses.append(avg_loss)
            LS1.append(avg_score1)
            LS2.append(avg_score2)
            LS3.append(avg_score3)
            reg_arr1.append(reg_loss1)
            reg_arr2.append(reg_loss2)
            reg_arr3.append(reg_loss3)
            
            # compute the F1 score between the ground truth features and the selected features
            f1_s1=0
            f1_s2=0
            f1_s3=0
            if feature_label1 is not None:
                f1_s1 = f1_score(feature_label1,alpha_p1>0.5)
                f1_s2 = f1_score(feature_label2,alpha_p2>0.5)
                f1_s3 = f1_score(feature_label3,alpha_p3>0.5)
            f1_score_list1.append(f1_s1)
            f1_score_list2.append(f1_s2)
            f1_score_list3.append(f1_s3)
             
            if (epoch+1) % self.display_step == 0:
                if feature_label1 is not None:
                    print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "score1=", "{:.9f}".format(avg_score1)\
                      , "score2=", "{:.9f}".format(avg_score2),"score3=", "{:.9f}".format(avg_score3), "reg1=", "{:.9f}".format(reg_fs1), "reg2=", "{:.9f}".format(reg_fs2), "reg3=", "{:.9f}".format(reg_fs3)
                      , "f1 - Mod1 = ","{:.4f}".format(f1_s1),"f1 - Mod2 = ","{:.4f}".format(f1_s2),"f1 - Mod3 = ","{:.4f}".format(f1_s3))

                else:
                    print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "score1=", "{:.9f}".format(avg_score1)\
                      , "score2=", "{:.9f}".format(avg_score2),"score3=", "{:.9f}".format(avg_score3), "reg1=", "{:.9f}".format(reg_fs1), "reg2=", "{:.9f}".format(reg_fs2),"reg3=", "{:.9f}".format(reg_fs3))
  
        print("Optimization Finished!")
        
        # compute the ls for each modality
        self.myker_joint = ker_joint
        self.mykern1 = kern1
        self.mykern2 = kern2
        self.mykern3 = kern3
        #self.myy = batch_ys
        self.myX1 = batch_xs1
        self.myX2 = batch_xs2
        self.myX3 = batch_xs3

        LS_list1 = []
        for i in range(self.myX1.shape[1]):
            LS_list1.append((self.myX1[:,i].reshape(1,-1)@self.myker_joint@self.myX1[:,i].reshape(-1,1))[0][0]/(np.linalg.norm(self.myX1[:,i])**2))
        
        LS_list2 = []
        for i in range(self.myX2.shape[1]):
            LS_list2.append((self.myX2[:,i].reshape(1,-1)@self.myker_joint@self.myX2[:,i].reshape(-1,1))[0][0]/(np.linalg.norm(self.myX2[:,i])**2))
        
        LS_list3 = []
        for i in range(self.myX3.shape[1]):
            LS_list3.append((self.myX3[:,i].reshape(1,-1)@self.myker_joint@self.myX3[:,i].reshape(-1,1))[0][0]/(np.linalg.norm(self.myX3[:,i])**2))

        
        return LS1,LS2,LS3,losses,reg_arr1,reg_arr2,reg_arr3,f1_score_list1,f1_score_list2,f1_score_list3,LS_list1,LS_list2,LS_list3

