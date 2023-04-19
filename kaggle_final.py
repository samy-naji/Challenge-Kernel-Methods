import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



from sklearn.metrics import accuracy_score

import seaborn as sns

from tqdm import tqdm

from scipy import optimize
from scipy.linalg import cho_factor, cho_solve

TRAINING_DATA=pd.read_pickle('./data-challenge-kernel-methods-2022-2023/training_data.pkl')
TRAINING_LABELS=pd.read_pickle('./data-challenge-kernel-methods-2022-2023/training_labels.pkl')
TEST_DATA=pd.read_pickle('./data-challenge-kernel-methods-2022-2023/test_data.pkl')

def from_graph_list_to_df(graph_list: list)->pd.DataFrame:


    dic_data={
    'Id':range(len(graph_list)),
    'max_depth':[],
    'avg_connectivity':[],
    'nb_nodes':[],
    'max_label':[],
    'min_label':[],
    'mean_label':[],
    'edge_0_rate':[],
    'edge_1_rate':[],
    'edge_2_rate':[],
    }
    for G in tqdm(graph_list):

        dic_data['max_depth'].append(max(list(nx.shortest_path_length(G,0).values())))
        dic_data['avg_connectivity'].append(nx.average_node_connectivity(G))
        dic_data['nb_nodes'].append(len(G.nodes))
        
        adj_matrix=np.zeros((len(G.nodes),len(G.nodes)))
        matrix_labels=[]
        edge_0,edge_1,edge_2=0.,0.,0.

        for i in range(len(G.nodes)):
            for j in dict(G.adj[i]).keys():
                weight=G.adj[i][j]['labels'][0]
                adj_matrix[i,j]=1
                if weight==0:
                    edge_0+=1
                elif weight==1:
                    edge_1+=1
                elif weight==2:
                    edge_2+=1
            matrix_labels.append(G.nodes[i]['labels'][0])
        nb_edges=adj_matrix.sum()/2.
        dic_data['edge_0_rate'].append(edge_0/nb_edges)
        dic_data['edge_1_rate'].append(edge_1/nb_edges)
        dic_data['edge_2_rate'].append(edge_2/nb_edges)
        dic_data['max_label'].append(max(matrix_labels))
        dic_data['min_label'].append(min(matrix_labels))
        dic_data['mean_label'].append(np.mean(matrix_labels))
    return pd.DataFrame(dic_data)





def geom_kernel_mano(X,Y,square:bool=False,lambd=0.1):
    res=np.zeros((len(X),len(Y)))
    if square:
        for i in tqdm(range(len(X))):
            for j in range(i+1):
                graph=nx.tensor_product(X[i],Y[j])
                adj_matrix=np.zeros((len(graph.nodes),len(graph.nodes)))

                for k in range(len(graph.nodes)):
                    for l in dict(graph.adj[list(graph.nodes)[k]]).keys():
                        adj_matrix[k,l]=1
                unit=np.ones(len(graph.nodes))
                res[i,j]=(unit.T).dot(np.linalg.inv(np.eye(adj_matrix.shape[0])-lambd*adj_matrix).dot(unit))
                if i==j:
                    res[i,i]=res[i,i]/2
        return res+res.T
    else:
    
        for i in tqdm(range(len(X))):
            for j in range(len(Y)):
                graph=nx.tensor_product(X[i],Y[j])
                adj_matrix=np.zeros((len(graph.nodes),len(graph.nodes)))

                for k in range(len(graph.nodes)):
                    for l in dict(graph.adj[list(graph.nodes)[k]]).keys():
                        adj_matrix[k,l]=1
                unit=np.ones(len(graph.nodes))
                res[i,j]=(unit.T).dot(np.linalg.inv(np.eye(adj_matrix.shape[0])-lambd*adj_matrix).dot(unit))
        return res



class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K=self.kernel(X,X,square=True)
        
        # Lagrange dual problem
        def loss(alpha):

            return  alpha.dot(K.dot(alpha))-2*alpha.dot(y)

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            
            return 2*K.dot(alpha)-2*y


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_ineq = lambda alpha: alpha*y  # '''---------------function defining the inequality constraint-------------------'''     
        jac_ineq = lambda alpha:  np.diag(y) # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        
        constraints = {
                        'type': 'ineq', 
                        'fun': fun_ineq,
                        'jac':jac_ineq}
        
        bounds=optimize.Bounds(np.zeros(N),self.C*np.ones(N))

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints,
                                   bounds=bounds)
        self.alpha = optRes.x

        ## Assign the required attributes

        ind=np.where(self.alpha >0)[0]
        self.support = X[ind] #'''------------------- A matrix with each row corresponding to a point that falls on the margin ------------------'''
        self.y_support = y[ind]

        f_X=np.linalg.multi_dot([K,np.diag(y),self.alpha])
        self.b = np.median(np.reciprocal(y)-f_X) #''' -----------------offset of the classifier------------------ '''
        self.norm_f = np.sqrt(self.alpha.dot(K.dot(self.alpha)))# '''------------------------RKHS norm of the function f ------------------------------'''


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N

        K=self.kernel(self.support,x)
        ind=np.where(0<self.alpha)[0]
        alpha_s=self.alpha[ind]
        return np.linalg.multi_dot([K.T, np.diag(self.y_support), alpha_s ])
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return  (d+self.b> 0)*1

