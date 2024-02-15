# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 00:35:55 2022

@author: lucam
"""

import torch
import numpy as np
from torch import nn
from torch import optim
import os 
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler2 = MinMaxScaler()
from scipy import signal
import matplotlib.pyplot as plt
def Mackey_glass(sample):
    file =r'/rdsgpfs/general/user/kjs18/home/Reservoir_computing/mackey_glass_t17.npy'
    data = np.load(file)
    fields = []
    for i in range(int(len(data)/sample)):
        fields.append(data[sample*i])
    return fields
def k_1(fields):
    fields2 = []
    fields2.append(fields[0])
    fields2.append(fields[1])
    for i in range(2,len(fields)):
        new_field = 0.4*fields[i-1]+0.4*fields[i-1]*fields[i-2]+0.6*fields[i]**3 + 0.1
        #new_field = fields[i]**2
        fields2.append(new_field)
    return(fields2)
def square(npoints,period):
    fields = []
    for i in range(npoints):
        #fields.append((np.sin(i*2*3.1415*n/(period))**m)*0.5*(max_field-min_field)+0.5*(min_field+max_field))
        #fields.append(-1*signal.sawtooth(i*2*3.1415*n/period)*0.5*(max_field-min_field)+0.5*(min_field+max_field))
        fields.append((signal.square(i*2*3.1415/period)))
    return fields
def sin(npoints,period, power):
    fields = []
    for i in range(npoints):
        fields.append((np.sin(i*2*3.1415/(period))**power))
        #fields.append(-1*signal.sawtooth(i*2*3.1415*n/period)*0.5*(max_field-min_field)+0.5*(min_field+max_field))
        #fields.append((signal.square(i*2*3.1415/period)))
    return fields

def cos(npoints,period, power):
    fields = []
    for i in range(npoints):
        fields.append((np.cos(i*2*3.1415/(period))**power))
        #fields.append(-1*signal.sawtooth(i*2*3.1415*n/period)*0.5*(max_field-min_field)+0.5*(min_field+max_field))
        #fields.append((signal.square(i*2*3.1415/period)))
    return fields

def saw(npoints,period,power,inverse = False):
    fields = []
    for i in range(npoints):
        #fields.append((np.sin(i*2*3.1415/(period))**power))
        if inverse == False:
            fields.append(-1*signal.sawtooth(i*2*3.1415/period))
        if inverse == True:
            fields.append(signal.sawtooth(i*2*3.1415/period))
        #fields.append((signal.square(i*2*3.1415/period)))
    return fields
def SINE_TASKS(names_ret=None):
    data_all = []
    names = []
    sin_squared = []
    sin_cubed = []
    sin_data2 = np.asarray(sin(800,30,1)[:800])
    for i in range(len(sin_data2)):
        sin_squared.append(sin_data2[i]**2)
        sin_cubed.append(sin_data2[i]**3)
    scaler2 = MinMaxScaler(feature_range = [0,1])

    sin_s = scaler2.fit_transform(np.asarray(sin_data2).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_s)

    names.append('sin_s')
    cos_s = scaler2.fit_transform(np.asarray(cos(800,30,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(cos_s)
    names.append('cos_s')
    sin_squared = scaler2.fit_transform(np.asarray(sin_squared).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_squared)
    names.append('sin_squared')
    sin_cubed = scaler2.fit_transform(np.asarray(sin_cubed).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_cubed)
    names.append('sin_cubed')
    sin_k_1 = scaler2.fit_transform(np.asarray(k_1(sin_s)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_k_1)
    names.append('sin_k_1')
    sin_to_square = scaler2.fit_transform(np.asarray(square(800,30)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_square)
    names.append('sin_to_square')
    sin_to_saw =  scaler2.fit_transform(np.asarray(saw(800,30,1,inverse = False)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_saw)
    names.append('sin_to_saw')
    sin_to_inv_saw =  scaler2.fit_transform(np.asarray(saw(800,30,1,inverse =True)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_inv_saw)
    names.append('sin_to_inv_saw')

    sin_to_half_sin =  scaler2.fit_transform(np.asarray(np.abs(sin(800,60,1))).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_half_sin)
    names.append('sin_to_half_sin')
    sin_to_third_sin =  scaler2.fit_transform(np.asarray(np.abs(sin(800,90,1))).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_third_sin)
    names.append('sin_to_third_sin')
    sin_to_double_sin =  scaler2.fit_transform(np.asarray(sin(800,15,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_double_sin)
    names.append('sin_to_double_sin')
    sin_to_triple_sin =  scaler2.fit_transform(np.asarray(sin(800,10,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_triple_sin)
    names.append('sin_to_triple_sin')
    
    cos_to_half_cos =  scaler2.fit_transform(np.asarray(np.abs(cos(800,60,1))).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(cos_to_half_cos)
    names.append('cos_to_half_cos')
    cos_to_third_cos =  scaler2.fit_transform(np.asarray(cos(800,90,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(cos_to_third_cos)
    names.append('cos_to_third_cos')
    cos_to_double_cos =  scaler2.fit_transform(np.asarray(cos(800,15,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(cos_to_double_cos)
    names.append('cos_to_double_cos')
    cos_to_triple_cos =  scaler2.fit_transform(np.asarray(cos(800,10,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(cos_to_triple_cos)
    names.append('cos_to_triple_cos')
   

    sin_to_half_square =  scaler2.fit_transform(np.asarray(square(800,60)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_half_square)
    names.append('sin_to_half_square')
    sin_to_third_square =  scaler2.fit_transform(np.asarray(square(800,90)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_third_square)
    names.append('sin_to_third_square')
    sin_to_double_square =  scaler2.fit_transform(np.asarray(square(800,15)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_double_square)
    names.append('sin_to_double_square')
    sin_to_triple_square =  scaler2.fit_transform(np.asarray(square(800,10)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_triple_square)
    names.append('sin_to_triple_square')
   

    sin_to_half_saw =  scaler2.fit_transform(np.asarray(saw(800,60,1,inverse=False)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_half_saw)
    names.append('sin_to_half_saw')

    sin_to_third_saw =  scaler2.fit_transform(np.asarray(saw(800,90,1,inverse=False)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_third_saw)
    names.append('sin_to_third_saw')
    sin_to_double_saw =  scaler2.fit_transform(np.asarray(saw(800,15,1,inverse=False)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_double_saw)
    names.append('sin_to_double_saw')
    sin_to_triple_saw =  scaler2.fit_transform(np.asarray(saw(800,10,1,inverse=False)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_triple_saw)
    names.append('sin_to_triple_saw')
   
    data_all_final = []
    for i in range(len(data_all)):
        data_all_final.append(data_all[i])

    for i in range(len(data_all)):
        for j in range(len(data_all)):
            data_all_final.append(scaler2.fit_transform(np.asarray(data_all[i][:len(data_all[j])]+data_all[j][:len(data_all[i])]).reshape(-1,1)).reshape(1,-1)[0])
            names.append(names[i]+'+'+names[j])

    data_sub_list = []
    for i in range(len(data_all)):
        for j in range(len(data_all)):
            data_all_final.append(scaler2.fit_transform(np.asarray(data_all[i][:len(data_all[j])]-data_all[j][:len(data_all[i])]).reshape(-1,1)).reshape(1,-1)[0])
            names.append(names[i]+'-'+names[j])

    data_multiply_list = []

    for i in range(len(data_all)):
        for j in range(len(data_all)):
            data_all_final.append(scaler2.fit_transform(np.asarray(data_all[i][:len(data_all[j])]*data_all[j][:len(data_all[i])]).reshape(-1,1)).reshape(1,-1)[0])
            names.append(names[i]+'*'+names[j])     
    
    data_all_double_final = []
    for i in range(len(data_all_final)):
        if np.isnan(data_all_final[i]).any()==False:
            if np.isinf(data_all_final[i]).any()==False:
                data_all_double_final.append(data_all_final[i])
    if names_ret==True:
        return data_all_double_final, names
    else:
        return data_all_double_final
        
def SINE_TASKS_FINAL(names_ret=None):
    data_all = []
    names = []
    sin_squared = []
    sin_cubed = []
    sin_data2 = np.asarray(sin(800,30,1)[:800])
    for i in range(len(sin_data2)):
        sin_squared.append(sin_data2[i]**2)
        sin_cubed.append(sin_data2[i]**3)
    scaler2 = MinMaxScaler(feature_range = [0,1])

    sin_s = scaler2.fit_transform(np.asarray(sin_data2).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_s)

    names.append('sin_s')

    sin_squared = scaler2.fit_transform(np.asarray(sin_squared).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_squared)
    names.append('sin_squared')
    sin_cubed = scaler2.fit_transform(np.asarray(sin_cubed).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_cubed)
    names.append('sin_cubed')
    


    sin_to_half_sin =  scaler2.fit_transform(np.asarray(np.abs(sin(800,60,1))).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_half_sin)
    names.append('sin_to_half_sin')

    sin_to_double_sin =  scaler2.fit_transform(np.asarray(sin(800,15,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_double_sin)
    names.append('sin_to_double_sin')
    sin_to_triple_sin =  scaler2.fit_transform(np.asarray(sin(800,10,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_triple_sin)
    names.append('sin_to_triple_sin')
    
    cos_s = scaler2.fit_transform(np.asarray(cos(800,30,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(cos_s)
    names.append('cos_s')
    cos_to_half_cos =  scaler2.fit_transform(np.asarray(np.abs(cos(800,60,1))).reshape(-1,1)).reshape(1,-1)[0]
    names.append('cos_to_half_cos')
    data_all.append(cos_to_half_cos)

    cos_to_double_cos =  scaler2.fit_transform(np.asarray(cos(800,15,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(cos_to_double_cos)
    names.append('cos_to_double_cos')
    cos_to_triple_cos =  scaler2.fit_transform(np.asarray(cos(800,10,1)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(cos_to_triple_cos)
    names.append('cos_to_triple_cos')
   
    sin_to_saw =  scaler2.fit_transform(np.asarray(saw(800,30,1,inverse = False)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_saw)
    names.append('sin_to_saw')
 


    #sin_to_half_saw =  scaler2.fit_transform(np.asarray(saw(800,60,1,inverse=False)).reshape(-1,1)).reshape(1,-1)[0]
    #data_all.append(sin_to_half_saw)
    #names.append('sin_to_half_saw')
 
    sin_to_double_saw =  scaler2.fit_transform(np.asarray(saw(800,15,1,inverse=False)).reshape(-1,1)).reshape(1,-1)[0]
    data_all.append(sin_to_double_saw)
    names.append('sin_to_double_saw')

   
    data_all_final = []
    for i in range(len(data_all)):
        data_all_final.append(data_all[i])

    for i in range(len(data_all)):
        for j in range(len(data_all)):
            data_all_final.append(scaler2.fit_transform(np.asarray(data_all[i][:len(data_all[j])]+data_all[j][:len(data_all[i])]).reshape(-1,1)).reshape(1,-1)[0])
            names.append(names[i]+'+'+names[j])

    #data_sub_list = []
    #for i in range(len(data_all)):
    #    for j in range(len(data_all)):
    #        data_all_final.append(scaler2.fit_transform(np.asarray(data_all[i][:len(data_all[j])]-data_all[j][:len(data_all[i])]).reshape(-1,1)).reshape(1,-1)[0])
    #        names.append(names[i]+'-'+names[j])

    data_multiply_list = []

    for i in range(len(data_all)):
        for j in range(len(data_all)):
            data_all_final.append(scaler2.fit_transform(np.asarray(data_all[i][:len(data_all[j])]*data_all[j][:len(data_all[i])]).reshape(-1,1)).reshape(1,-1)[0])
            names.append(names[i]+'*'+names[j])     
    
 
    if names_ret==True:
        return data_all_final, names
    else:
        return data_all_final       
def Narma(s, step, start):

    
    T=np.shape(s)[0]
    y=torch.zeros([T])
    
    alpha=0.3
    beta=0.01
    gamma=2
    delta=0.1
    
    for t in range(step,T):
        
        y[t]=alpha*y[t-1]+beta*y[t-1]*torch.sum(y[t-step:t])+gamma*s[t-step]*s[t-1]+delta
        
            
    return y




class ESN:
    
    def __init__(self,N,N_in,N_av,alpha,rho,gamma):        
        
        self.N=N
        self.alpha=alpha
        self.rho=rho
        self.N_av=N_av
        self.N_in=N_in
        self.gamma=gamma
        
        diluition=1-N_av/N
        W=np.random.uniform(-1,1,[N,N])
        W=W*(np.random.uniform(0,1,[N,N])>diluition)
        eig=np.linalg.eigvals(W)
        self.W=torch.from_numpy(self.rho*W/(np.max(np.absolute(eig)))).float()
        
        
        self.x=[]
        
        if self.N_in==1:
            
            self.W_in=2*np.random.randint(0,2,[self.N_in,self.N])-1
            self.W_in=torch.from_numpy(self.W_in*self.gamma).float()
            
            
        else:
            
            self.W_in=np.random.randn(self.N_in,self.N)
            self.W_in=torch.from_numpy(self.gamma*self.W_in).float()
        
        
    def Reset(self,s):
        
        batch_size=np.shape(s)[0]
        self.x=torch.zeros([batch_size,self.N])
        
    def ESN_1step(self,s):
        
        self.x=(1-self.alpha)*self.x+self.alpha*torch.tanh(torch.matmul(s,self.W_in)+torch.matmul(self.x,self.W))
        
    def ESN_response(self,Input):
        
        T=Input.shape[2]
        X=torch.zeros(Input.shape[0],self.N,T)
        
        self.Reset(Input[:,0])
        
        for t in range(T):
            
            self.ESN_1step(Input[:,:,t])
            X[:,:,t]=torch.clone(self.x)
            
        return X


class DATA_CLEAN:
    
    def __init__(self,targets,data,p_te):
        
        T_data=250
        X = data
        Y = targets
        #print(data.shapoe)
        #self.Task=Task
        self.T_data = T_data
        self.X_original=X
        self.Y_original=Y
        
        self.X=X
        self.Y=Y
        print(self.X.shape)
        self.X_tr=[]
        self.X_val=[]
        self.X_te=[]
        
        self.X_tr_M=[]
        self.X_val_M=[]
        self.X_te_M=[]
        
        self.Y_tr=[]
        self.Y_val=[]
        self.Y_te=[]
        
        self.it_ind=[]
        
        self.Corr=[]
        
        self.Cov=[]
        self.w=[]
        self.v=[]
        
        self.p_te=p_te
        
        self.index_tr=[]
        self.index_val=[]
        self.index_te=[]
        
        self.Z_val=[]
        self.Z_te=[]
        
    def CrossVal(self,te_ind,val_ind):
        # This code splits the data
        N_data=np.shape(self.X)[0]
        
        N_trans=20
        
        N_te=int(np.floor((N_data-N_trans)*self.p_te))
        
        N_val=int(np.floor((N_data-N_trans-N_te)*0.1))
        
        index=np.arange(N_trans,N_data)
        
        index_te=np.copy(index[te_ind*N_te:(te_ind+1)*N_te])
        index_noT=np.delete(index,index_te-N_trans)
        
        index_val=np.copy(index_noT[val_ind*N_val:(val_ind+1)*N_val])
        
        index_tr=np.delete(index,np.concatenate([index_te-N_trans,index_val-N_trans],0))
        
        
        X_tr=np.copy(self.X[index_tr,:])
        
        X_M=np.max(np.abs(X_tr),0)
        
        X_tr=X_tr/np.tile(np.expand_dims(X_M,0),[np.shape(X_tr)[0],1])
        X_val=np.copy(self.X[index_val,:])/np.tile(np.expand_dims(X_M,0),[N_val,1])
        X_te=np.copy(self.X[index_te,:])/np.tile(np.expand_dims(X_M,0),[N_te,1])
        
        Y_tr=np.copy(self.Y[index_tr,:])
        Y_val=np.copy(self.Y[index_val,:])
        Y_te=np.copy(self.Y[index_te,:])
        
        self.X_tr_M=np.copy(X_tr)
        self.X_val_M=np.copy(X_val)
        self.X_te_M=np.copy(X_te)
        
        self.Y_tr_M=np.copy(Y_tr)
        self.Y_val_M=np.copy(Y_val)
        self.Y_te_M=np.copy(Y_te)
        
        self.X_tr=torch.from_numpy(X_tr).float()
        self.X_te=torch.from_numpy(X_te).float()
        self.X_val=torch.from_numpy(X_val).float()
        
        self.Y_tr=torch.from_numpy(Y_tr).float()
        self.Y_te=torch.from_numpy(Y_te).float()
        self.Y_val=torch.from_numpy(Y_val).float()
        
        self.index_val=index_val
        self.index_te=index_te
        
        
    def CORR(self):
        
        Corr=np.corrcoef(np.transpose(self.X_tr_M))
        
        self.Corr=Corr
        
        
    def CORR_Analysis(self,th,removal):
        
        
        N=np.shape(self.Corr)[0]
        
        N_red=np.abs(self.Corr)>th
        
        RM=[]
        
        for i in range(N):
            
            if any(N_red[i,:]==1):
                
                
                ind=np.where(N_red[i,:]==1)[0]
                
                if any(ind>i):
                    
                    RM.append(i)
        
        RM=np.array(RM)
        
        
        if removal==True and RM!=[]:
                
            self.X_tr_M=np.delete(self.X_tr,RM,1)    
            self.X_val_M=np.delete(self.X_val,RM,1)
            self.X_te_M=np.delete(self.X_te,RM,1)
            
        else:
            
            self.X_tr_M=self.X_tr    
            self.X_val_M=self.X_val
            self.X_te_M=self.X_te
            
        
        return RM
    
    def Benchmark_Correlation(self,ths):
        
        ## Arange the thetas to loop through
        N_fit=np.shape(ths)[0]
        
        # Ridge parameter alphas
        alphas=np.array([1e-4,1e-3,1e-2,5*1e-2,1*1e-1])  
        # Number of alphas
        N_alpha=np.shape(alphas)[0]
        
        # Number of data points 
        N_out=np.shape(self.Y_tr)[1]
        
        # Number of splits in inner loop
        N_split=10
        
        # Number of outputs
        N=np.shape(self.X_tr_M)[1]
        
        # Number of test splits
        N_split_te=int(1/self.p_te)
        
        # Mask for the outputs
        RM=np.zeros([N_split_te,N])
        
        # Array to hold best hyperparameters
        Best_hyper=np.zeros([N_split_te,3])
        
        ## Arrays to hold the MSE's
        MSE_Te=np.zeros([N_split_te,N_out])
        MSE_Val=np.zeros([N_split_te])
        MSE_Tr=np.zeros([N_split_te])
        
        ## Arrays to hold the predictions
        Z_val=np.zeros([N_split_te,self.T_data,N_out,2])
        Z_te=np.zeros([N_split_te,self.T_data,N_out,2])
        
        ## Loop through number of test splits
        for l in range(N_split_te):
            
            # Arrays to MSE's and the validation output
            mse_Val=np.zeros([N_split,np.shape(ths)[0],N_alpha,N_out])
            mse_Tr=np.zeros([N_split,np.shape(ths)[0],N_alpha,N_out])
            O_val=np.zeros([N_fit,N_alpha,self.T_data,N_out,2])
        
            for j in range(N_split):
                
                # Split the data
                self.CrossVal(l,j)
                # Calculate the correlation matrix
                self.CORR()
                
                
                for i in range(N_fit):
                    # Remove the correlated features based on the chosen theta               
                    _=self.CORR_Analysis(ths[i],True)
                                    
                    # Loop through the alphas
                    for k in range(N_alpha):
                        # Define the ridge regression
                        model=Ridge(alpha = alphas[k],fit_intercept = False,copy_X = True)
                        
                        # Fit the model to the training data
                        reg = model.fit(self.X_tr_M, self.Y_tr_M)
                        
                        # Get the outputs for the train and validate sets
                        output_val = reg.predict(self.X_val_M)
                        output_train = reg.predict(self.X_tr_M)
                        
                        # Put the outputs and MSEs into the arrays
                        O_val[i,k,self.index_val,:,0]=np.copy(output_val)
                        O_val[i,k,self.index_val,:,1]=np.copy(self.Y_val_M)
                        
                        mse_Tr[j,i,k,:] = np.mean((self.Y_tr_M-output_train)**2,0)
                        mse_Val[j,i,k,:] = np.mean((self.Y_val_M-output_val)**2,0)
                        
            
            # Average the MSEs for the train and test across all of the splits
            M_Tr=np.mean(np.mean(mse_Tr,0),2) 
            M_Val=np.mean(np.mean(mse_Val,0),2) 
            
            # Get the index of the lowest MSE val to get best alpha and theta
            ind=np.argwhere(M_Val==np.min(M_Val))[0]
            
            # Put the MSEs in the overall array from the best alphas and th
            MSE_Val[l]=np.copy(M_Val[ind[0],ind[1]])
            MSE_Tr[l]=np.copy(M_Tr[ind[0],ind[1]])
            
            # Put the best validation outputs and targets in.
            Z_val[l,:,:,0]=np.copy(O_val[ind[0],ind[1],:,:,0])
            Z_val[l,:,:,1]=np.copy(O_val[ind[0],ind[1],:,:,1])
            
            ## New training set comprises the original train and validation
            self.X_tr_M=np.concatenate([self.X_tr,self.X_val],0)
            
            ## Do the correlation analysis
            self.CORR()
            
            # Sort the features and get the indicies of the ones removed
            EL_removed=self.CORR_Analysis(ths[ind[0]],True)
            
            # Save the removed feature indices
            if EL_removed!=[]:
                RM[l,EL_removed]=1
            
            
            # Get the best parameters
            Best_hyper[l,0]=ths[ind[0]] # Best theta
            Best_hyper[l,1]=alphas[ind[1]] # Best alpha
            Best_hyper[l,2]=np.sum(RM[l,:]==0) # Number of features used
            
            # Generate new training sets
            self.X_tr_M=np.concatenate([self.X_tr_M,self.X_val_M],0)
            self.Y_tr_M=np.concatenate([self.Y_tr_M,self.Y_val_M],0)
            
            # Fit the data using the best parameters and features
            model=Ridge(alpha = alphas[ind[1]],fit_intercept = False,copy_X = True)
            reg = model.fit(self.X_tr_M, self.Y_tr_M)
            
            # Get the output
            output_test = reg.predict(self.X_te_M)
            
            # Save the MSE for the test
            MSE_Te[l,:] = np.mean((self.Y_te_M-output_test)**2,0)
                
            
            Z_te[l,self.index_te,:,0]=np.copy(output_test)
            Z_te[l,self.index_te,:,1]=np.copy(self.Y_te_M)
            
        
        print('Performance: TE, ', np.mean(MSE_Te), 'VAL ', np.mean(MSE_Val), 'TR ', np.mean(MSE_Tr))

        
        return MSE_Tr, MSE_Val, MSE_Te, RM, Best_hyper, Z_val, Z_te
    
    def Measure_MC(self,ths):
        
        MSE_Tr, MSE_Val, MSE_Te, RM, Best_hyper, Z_val, Z_te=self.Benchmark_Correlation(ths)
        
        MC=np.zeros([np.shape(Z_te)[1]])
        
        for k in range(np.shape(Z_te)[1]):
            
            ind_te=np.where(Z_te[:,k,0]!=0)
            u_test=np.copy(Z_te[ind_te,k,0])
            u_pred=np.copy(Z_te[ind_te,k,1])
            
            MC[k]=( np.mean(u_test*u_pred) - np.mean(u_test)*np.mean(u_pred) )**2 / (np.var(u_test) * np.var(u_pred))
        
        
        print('MC: ', np.sum(MC))
        
        return MSE_Tr, MSE_Val, MSE_Te, RM, Best_hyper, Z_val, Z_te, MC
    
    
    def Measure_NL(self):
        
        k_min=5
        N_fit=self.k_max-k_min
        Ks=np.arange(k_min,self.k_max+1)

        alphas=np.array([1e-4,1e-3,1e-2,5*1e-2,1*1e-1])        
        N_alpha=np.shape(alphas)[0]
        
        N_out=np.shape(self.Y_tr)[1]
        
        N_split=10
        
        N_split_te=int(1/self.p_te)
        
        Best_hyper=np.zeros([N_split_te,3])
        
        MSE_Te=np.zeros([N_split_te,N_out])
        MSE_Val=np.zeros([N_split_te])
        MSE_Tr=np.zeros([N_split_te])
        
        Z_val=np.zeros([N_split_te,self.T_data,N_out,2])
        Z_te=np.zeros([self.T_data,N_out,2])
        
        for l in range(N_split_te):
            
            mse_Val=np.zeros([N_split,N_fit,N_alpha,N_out])
            mse_Tr=np.zeros([N_split,N_fit,N_alpha,N_out])
            O_val=np.zeros([N_fit,N_alpha,self.T_data,N_out,2])
        
            for j in range(N_split):
                
                self.CrossVal(l,j)
                
                for i in range(N_fit):
                                    
                    self.X_tr_M=np.copy(self.X_tr[:,0:Ks[i]])    
                    self.X_val_M=np.copy(self.X_val[:,0:Ks[i]])
                    self.X_te_M=np.copy(self.X_te[:,0:Ks[i]])
                                    
                    for k in range(N_alpha):
                
                        model=Ridge(alpha = alphas[k],fit_intercept = False,copy_X = True)
                        reg = model.fit(self.X_tr_M, self.Y_tr_M)
            
                        output_val = reg.predict(self.X_val_M)
                        output_train = reg.predict(self.X_tr_M)
                        O_val[i,k,self.index_val,:,0]=np.copy(output_val)
                        O_val[i,k,self.index_val,:,1]=np.copy(self.Y_val_M)
                        
                        mse_Tr[j,i,k,:] = np.mean((self.Y_tr_M-output_train)**2,0)
                        mse_Val[j,i,k,:] = np.mean((self.Y_val_M-output_val)**2,0)
                        
                  
            M_Tr=np.mean(np.mean(mse_Tr,0),2) 
            M_Val=np.mean(np.mean(mse_Val,0),2) 
            
            
            ind=np.argwhere(M_Val==np.min(M_Val))[0]
            
            MSE_Val[l]=np.copy(M_Val[ind[0],ind[1]])
            MSE_Tr[l]=np.copy(M_Tr[ind[0],ind[1]])
    
            Z_val[l,:,:,0]=np.copy(O_val[ind[0],ind[1],:,:,0])
            Z_val[l,:,:,1]=np.copy(O_val[ind[0],ind[1],:,:,1])
            
            self.X_tr_M=np.concatenate([self.X_tr,self.X_val],0)
            self.X_tr_M=self.X_tr_M[:,0:Ks[ind[0]]]
            self.X_te_M=self.X_te_M[:,0:Ks[ind[0]]]
            
            self.Y_tr_M=np.concatenate([self.Y_tr_M,self.Y_val_M],0)
            
            
            Best_hyper[l,0]=Ks[ind[0]]
            Best_hyper[l,1]=alphas[ind[1]]

            model=Ridge(alpha = alphas[ind[1]],fit_intercept = False,copy_X = True)
            reg = model.fit(self.X_tr_M, self.Y_tr_M)
            
            output_test = reg.predict(self.X_te_M)
            
            MSE_Te[l,:] = np.mean((self.Y_te_M-output_test)**2,0)
                
            
            Z_te[self.index_te,:,0]=np.copy(output_test)
            Z_te[self.index_te,:,1]=np.copy(self.Y_te_M)
            
        
        print('Performance: TE, ', np.mean(MSE_Te), 'VAL ', np.mean(MSE_Val), 'TR ', np.mean(MSE_Tr))
        
        MC=np.zeros([np.shape(Z_te)[1]])
        
        for k in range(np.shape(Z_te)[1]):
            
            ind_te=np.where(Z_te[:,k,0]!=0)
            u_test=np.copy(Z_te[ind_te,k,0])
            u_pred=np.copy(Z_te[ind_te,k,1])
            
            MC[k]=( np.mean(u_test*u_pred) - np.mean(u_test)*np.mean(u_pred) )**2 / (np.var(u_test) * np.var(u_pred))
        
        
        NL=1-np.mean(MC)
        
        print('NL: ', NL)
        return MSE_Tr, MSE_Val, MSE_Te, Best_hyper, Z_val, Z_te, MC
        
    
    
    def Evolutionary(self, Feats, p_mutation, N_iteration, alphas):
        
        Parents_in=False
        
        # Define the parameters for the algorithm
        N_children=200
        N_split=10
        N_best=10
        
        # Number of split featuers
        N_split_te=int(1/self.p_te)
        
        N_out=np.shape(self.Y)[1]
        N_feat=np.shape(self.X)[1]
            
        MSE1=np.zeros([N_split_te,N_iteration,N_best,N_out])
        MSE2=np.zeros([N_split_te,N_iteration,N_best,N_out])
        
        Z_val=np.zeros([N_split_te,self.T_data,N_out,3])
        Z_te=np.zeros([self.T_data,N_out,2])
        
        MSE=[]
        MSE_bench=[]
        
        for split in range(N_split_te):
            
            Mask=np.copy(Feats[split,:])
            
            alpha=np.copy(alphas[split])
            
            model=Ridge(alpha = alpha,fit_intercept = False,copy_X = True)
            best_ch=[]

            self.CrossVal(split,9)

            N_te=np.shape(self.X_te)[0]
            N_te=int(N_te/2)
            
            index_te1=self.index_te[0:N_te]
            index_te2=self.index_te[N_te:]
            
            X_te1=np.copy(self.X_te[0:N_te,:])
            Y_te1=np.copy(self.Y_te[0:N_te,:])

            X_te2=np.copy(self.X_te[N_te:,:])
            Y_te2=np.copy(self.Y_te[N_te:,:])

            MSE_te1=np.zeros([N_iteration,N_best])
            MSE_te2=np.zeros([N_iteration,N_best])

            RM=np.where(Mask==1)

            X_te1_M=np.delete(X_te1,RM,1)
            X_te2_M=np.delete(X_te2,RM,1)

            self.X_tr_M=np.delete(np.concatenate([self.X_tr,self.X_val],0),RM,1)
            self.Y_tr_M=np.concatenate([self.Y_tr,self.Y_val],0)

            reg=model.fit(self.X_tr_M, self.Y_tr_M)

            MSE_bench.append(np.mean((Y_te1-reg.predict(X_te1_M))**2))
            MSE_bench.append(np.mean((Y_te2-reg.predict(X_te2_M))**2))
            
            print('Split Number ', split)
            print('Before Evolution: ', MSE_bench[-2],MSE_bench[-1])
            
            if Parents_in:

                N_loop=N_children+N_best

            if not(Parents_in):
                
                N_loop=N_children
            
            
            O_val=np.zeros([N_iteration,N_loop,self.T_data,N_out,2])
            O_te=np.zeros([N_iteration,N_best,self.T_data,N_out,2])
            
            for n in range(N_iteration):

                if n==0:    

                    if Parents_in:

                        Masks=np.tile(np.expand_dims(Mask,1),[1,N_children+N_best])

                    if not(Parents_in):

                        Masks=np.tile(np.expand_dims(Mask,1),[1,N_children])


                else:

                    Parents=np.copy(Masks[:,best_ch])

                    ind_mix=np.random.permutation(N_best)
                    Masks=np.zeros([N_feat,N_children])
                    N_mix=int(2*N_children/N_best)

                    for l in range(int(N_best/2)):

                        cross=np.random.rand(N_feat,N_mix)<0.5

                        Masks[:,N_mix*l:N_mix*(l+1)]=np.tile(np.expand_dims(Parents[:,ind_mix[2*l]],1),[1,N_mix])*cross+np.tile(np.expand_dims(Parents[:,ind_mix[2*l+1]],1),[1,N_mix])*(1-cross)


                    if Parents_in:

                        Masks=np.concatenate([Masks,Parents],1)


                Mutations=np.random.rand(N_feat,N_children)<p_mutation

                if Parents_in:

                    Masks[:,0:N_children]=Masks[:,0:N_children]*(1-Mutations)+(1-Masks[:,0:N_children])*Mutations

                if not(Parents_in):
                    
                    Masks=Masks*(1-Mutations)+(1-Masks)*Mutations

                MSE_val=np.zeros([N_loop,N_split])
                                
                for i in range(N_loop):

                    RM=np.where(Masks[:,i]==1)


                    for j in range(N_split):

                        self.CrossVal(split,j)

                        self.X_tr_M=np.delete(self.X_tr,RM,1)    
                        self.X_val_M=np.delete(self.X_val,RM,1)

                        reg=model.fit(self.X_tr_M, self.Y_tr_M)

                        MSE_val[i,j]=np.mean((self.Y_val_M-reg.predict(self.X_val_M))**2)
                        
                        O_val[n,i,self.index_val,:,0]= reg.predict(self.X_val_M)
                        O_val[n,i,self.index_val,:,1]= np.copy(self.Y_val_M)
                                                
                        
                best_ch=np.argsort(np.mean(MSE_val,1))[0:N_best]
                
                O_val[:,0:N_best,:,:]=O_val[:,best_ch,:,:,:]
                
                for i in range(N_best):

                    RM=np.where(Masks[:,best_ch[i]]==1)

                    self.X_tr_M=np.delete(np.concatenate([self.X_tr,self.X_val],0),RM,1)
                    self.Y_tr_M=np.concatenate([self.Y_tr,self.Y_val],0)

                    reg=model.fit(self.X_tr_M, self.Y_tr_M)

                    X_te1_M=np.delete(X_te1,RM,1)
                    X_te2_M=np.delete(X_te2,RM,1)
                    
                    MSE1[split,n,i,:]=np.mean((Y_te1-reg.predict(X_te1_M))**2,0)
                    MSE2[split,n,i,:]=np.mean((Y_te2-reg.predict(X_te2_M))**2,0)
                    
                    MSE_te1[n,i]=np.mean(MSE1[split,n,i,:])
                    MSE_te2[n,i]=np.mean(MSE2[split,n,i,:])
                    
                    O_te[n,i,index_te1,:,0]= reg.predict(X_te1_M)
                    O_te[n,i,index_te2,:,0]= reg.predict(X_te2_M)
                    O_te[n,i,index_te1,:,1]= np.copy(Y_te1)
                    O_te[n,i,index_te2,:,1]= np.copy(Y_te2)
                    
            O_val=O_val[:,0:N_best,:,:]
            
            ind1=np.argwhere(MSE_te1==np.min(MSE_te1))[0]
            ind2=np.argwhere(MSE_te2==np.min(MSE_te2))[0]
            
            Z_val[split,:,:,0]=np.copy(O_val[ind2[0],ind2[1],:,:,0])
            Z_val[split,:,:,1]=np.copy(O_val[ind1[0],ind1[1],:,:,0])
            Z_val[split,:,:,2]=np.copy(O_val[ind1[0],ind1[1],:,:,1])
            
            Z_te[index_te1,:,0]=O_te[ind2[0],ind2[1],index_te1,:,0]
            Z_te[index_te2,:,0]=O_te[ind1[0],ind1[1],index_te2,:,0]
            Z_te[self.index_te,:,1]=O_te[ind1[0],ind1[1],self.index_te,:,1]
            
            
            MSE.append(MSE1[split,ind2[0],ind2[1],:])
            MSE.append(MSE2[split,ind1[0],ind1[1],:])
            
            print('After Evolution: ', MSE_te1[ind2[0],ind2[1]],MSE_te2[ind1[0],ind1[1]], ind1[0], ind2[0])
            print('Minima (Overfitting): ', np.min(MSE_te1),np.min(MSE_te2))
        
        MSE=np.array(MSE)
        MSE_bench=np.array(MSE_bench)
        
        print('Overall, Before and After: ', np.mean(MSE_bench), np.mean(MSE))
            
        return MSE, MSE_bench, Z_val, Z_te, RM

class DATA:
    
    def __init__(self,targets,datafolder,p_te,option,Task,Task_list_inp=None):    
        T_data=250
        self.T_data=T_data
        
        if option !='all' and option[0:3]!='ESN' and Task!='SINE_NP' and Task!='SINE_NP_KS' and Task!='SINE_NP_KS_REDUCE' and 'SINE' not in Task:
        
            if option[0]==0:
                X1=np.load(os.path.join(datafolder,'0.ASVI04_From_ASVI04_ASVI04_Original.npy'),allow_pickle=True)

            if option[0]==7:
                X1=np.load(os.path.join(datafolder,'7.ASVI04_From_HDS_GS_HDS_GS_Original.npy'),allow_pickle=True)

            if option[0]==14:
                X1=np.load(os.path.join(datafolder,'14.HDS_GS_From_Pinwheel_Pinwheel.npy'),allow_pickle=True)


            X1=X1[0:T_data,:]

            if np.shape(option)[0]==1:

                X=np.copy(X1)


            if np.shape(option)[0]>1:

                n_d=len(str(option[1]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[1])+'.'):

                            X2=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X2=X2[0:T_data,:]


                            if np.shape(option)[0]==2:

                                X=np.concatenate([X1,X2],1)


            if np.shape(option)[0]>2:

                n_d=len(str(option[2]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[2])+'.'):

                            X3=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X3=X3[0:T_data,:]

                            X=np.concatenate([X1,X2,X3],1)

        
        if option=='all':
            X=np.zeros([T_data,1])
            print('here')
            for file in os.listdir(datafolder):
                
                if '.npy' in file: 
                    
                    X1=np.load(os.path.join(datafolder,file),allow_pickle=True)
                    if np.shape(X1)[0]>=T_data and file !='all_targets.npy' and file!='target_names.npy':
                        
                        X=np.concatenate([X,X1[0:T_data,:]],1)  

            X=X[:,1:]
            
        
        if option[0:3]=='ESN':
            
            X=np.load(os.path.join(datafolder,option+'.npy'),allow_pickle=True)
        
        if Task!='SINE_NP' and Task!='SINE_NP_KS' and Task!='SINE_NP_KS_REDUCE' and 'SINE' not in Task and Task!='SINE_NP_KS_REDUCE_FINAL':
            targets= np.load(os.path.join(datafolder,'all_targets.npy'),allow_pickle = True)     
            names = np.load(os.path.join(datafolder,'target_names.npy'),allow_pickle = True)
        
        
        
            task_list = []
            N_task=30

            for i in range(N_task):
                task_list.append('MG '+str(i))
            
            
            count=0
            Y=np.zeros([T_data,N_task])
            
            for task in task_list:
                
                
                names2 = names.tolist()
                ID = names2.index(task)

                target = targets[ID]
                
                Y[:,count]=np.copy(target[0:T_data])
                              
                count=count+1
                
            if Task=='Prediction_Narma':
                
                Y_new=np.zeros([np.shape(Y)[0],np.shape(Y)[1]])
                start=15
                n_Narma=7
                
                for t in range(np.shape(Y)[1]):
                    
                    Y_new[:,t]=Narma(Y[:,t],n_Narma,start)
                
                
                Y=np.copy(Y_new)
            
            
            if Task=='Narma':
                
                N_narma=15
                
                signal=np.copy(Y[:,0]);
                Y=np.zeros([T_data,N_narma])    
                
                start=15
                
                for n in range(N_narma):
                    
                    Y[:,n]=Narma(signal,n,start)
                
             
            if Task=='MC':
                
                k_max=8
                self.k_max=k_max
                signal=np.copy(Y[:,0]);
                
                Y=np.zeros([T_data-k_max,k_max])
                
                for k in range(k_max):
                    
                    Y[:,k]=signal[k_max-k-1:-k-1]
                
                X=np.copy(X[k_max:,:])
                
                
            if Task=='NL':
                
                k_max=8
                self.k_max=k_max
                signal=np.copy(Y[:,0]);
                Y=np.zeros([T_data-k_max,k_max])
                
                for k in range(k_max):
                    if k==0:
                        Y[:,k]=signal[k_max-k:]
                    else:
                        Y[:,k]=signal[k_max-k:-k]
                    
                X=np.copy(X[k_max:,1:])
                
                Var=np.var(X,0)
                Ch=np.argsort(Var)
                
                Ch=Ch[np.shape(Ch)[0]-60:]

                X=np.copy(X[:,Ch])
                
                X_M=np.max(np.abs(X),0)
                X=X/np.tile(np.expand_dims(X_M,0),[np.shape(X)[0],1])
                
                swap=np.copy(Y)
                Y=np.copy(X)
                X=np.copy(swap)
        
        if Task=='SINE_NARMA':
            if option[0]==0:
                X1=np.load(os.path.join(datafolder,'0.HDS_GS_Original.npy'),allow_pickle=True)

            if option[0]==1:
                X1=np.load(os.path.join(datafolder,'1.ASVI04_original.npy'),allow_pickle=True)

            if option[0]==2:
                X1=np.load(os.path.join(datafolder,'2.ASVI07_Original.npy'),allow_pickle=True)
            if option[0]==3:
                X1=np.load(os.path.join(datafolder,'3.Pinwheel_ASVI_y.npy'),allow_pickle=True)
            if option[0]==4:
                X1=np.load(os.path.join(datafolder,'4.Pinwheel_ASVI_x.npy'),allow_pickle=True)


            X1=X1[0:T_data,:]

            if np.shape(option)[0]==1:

                X=np.copy(X1)


            if np.shape(option)[0]>1:

                n_d=len(str(option[1]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[1])+'.'):

                            X2=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X2=X2[0:T_data,:]


                            if np.shape(option)[0]==2:

                                X=np.concatenate([X1,X2],1)


            if np.shape(option)[0]>2:

                n_d=len(str(option[2]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[2])+'.'):

                            X3=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X3=X3[0:T_data,:]

                            X=np.concatenate([X1,X2,X3],1)

            sin_data2 = np.asarray(sin(800,30,1)[:T_data])
            S = scaler2.fit_transform(np.asarray(sin_data2).reshape(-1,1)).reshape(1,-1)[0]

            N_narma=15
                
            #signal=np.copy(Y[:,0]);
            Y=np.zeros([T_data,N_narma])    
            
            start=15
            
            for n in range(N_narma):
                
                Y[:,n]=Narma(S,n,start)

        if Task=='SINE_NP':
            
            if option[0]==0:
                X1=np.load(os.path.join(datafolder,'0.HDS_GS_Original.npy'),allow_pickle=True)

            if option[0]==1:
                X1=np.load(os.path.join(datafolder,'1.ASVI04_original.npy'),allow_pickle=True)

            if option[0]==2:
                X1=np.load(os.path.join(datafolder,'2.ASVI07_Original.npy'),allow_pickle=True)
            if option[0]==3:
                X1=np.load(os.path.join(datafolder,'3.Pinwheel_ASVI_y.npy'),allow_pickle=True)
            if option[0]==4:
                X1=np.load(os.path.join(datafolder,'4.Pinwheel_ASVI_x.npy'),allow_pickle=True)


            X1=X1[0:T_data,:]

            if np.shape(option)[0]==1:

                X=np.copy(X1)


            if np.shape(option)[0]>1:

                n_d=len(str(option[1]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[1])+'.'):

                            X2=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X2=X2[0:T_data,:]


                            if np.shape(option)[0]==2:

                                X=np.concatenate([X1,X2],1)


            if np.shape(option)[0]>2:

                n_d=len(str(option[2]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[2])+'.'):

                            X3=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X3=X3[0:T_data,:]

                            X=np.concatenate([X1,X2,X3],1)
            plt.plot(X[0])
            plt.show()
            sin_data2 = np.asarray(sin(800,30,1)[:800])
            S = scaler2.fit_transform(np.asarray(sin_data2).reshape(-1,1)).reshape(1,-1)[0]
            S=2*(S-np.min(S))/(np.max(S-np.min(S)))-1
            S1=S[0:T_data]
            
            C1=np.zeros([T_data])
            C1[0]=np.sqrt(1-S1[0]**2)
            C1[-1]=np.sqrt(1-S1[-1]**2)
            C1[1:-1]=np.sign( (S1[2:]-S1[1:-1])/2+(S1[1:-1]-S1[0:-2])/2 )*np.sqrt(1-S1[1:-1]**2)
            
            S1_2=np.zeros([T_data])
            count=0
            S1_2=np.sqrt((1-C1)/2)
            Sg=np.zeros([T_data])
            Sg[1:-1]=np.sign( (S1_2[2:]-S1_2[1:-1])/2+(S1_2[1:-1]-S1_2[0:-2])/2 )
            Sg[0]=1
            Sg[-1]=1
            P=np.arange(0,T_data)
            P=np.reshape(P,[int(T_data/2),2])
            P=np.reshape(np.delete(P,np.arange(0,int(np.shape(P)[0]/2))*2,0),[-1])

            for t in range(1,T_data):

                if Sg[t]*Sg[t-1]<0:
                    count=count+1

                if np.any(P==count):

                    S1_2[t]=-np.sqrt((1-C1[t])/2)

                else:

                    S1_2[t]=np.sqrt((1-C1[t])/2)
            
            C1_2=np.zeros([T_data])
            C1_2[0]=np.sqrt(1-S1_2[0]**2)
            C1_2[-1]=np.sqrt(1-S1_2[-1]**2)
            C1_2[1:-1]=np.sign( (S1_2[2:]-S1_2[1:-1])/2+(S1_2[1:-1]-S1_2[0:-2])/2 )*np.sqrt(1-S1_2[1:-1]**2)

            S1_3=np.sqrt((1-C1_2)/2)
            
            C1_3=np.zeros([T_data])
            C1_3[0]=np.sqrt(1-S1_3[0]**2)
            C1_3[-1]=np.sqrt(1-S1_3[-1]**2)
            C1_3[1:-1]=np.sign( (S1_3[2:]-S1_3[1:-1])/2+(S1_3[1:-1]-S1_3[0:-2])/2 )*np.sqrt(1-S1_3[1:-1]**2)
            
            C2=C1**2-S1**2
            S2=2*S1*C1
            S3=3*S1-4*S1**3
            C3=4*C1**3-3*C1
            
            Y=np.zeros([T_data,31])
            
            Y[:,0]=S1
            Y[:,1]=S2
            Y[:,2]=S3
            Y[:,3]=C1
            Y[:,4]=C2
            Y[:,5]=np.absolute(S1_2)
            Y[:,6]=np.absolute(C1_2)
            
            Y[:,7]=S1+C2
            Y[:,8]=C1+S2
            Y[:,9]=C1-S2
            
            Y[:,10]=np.absolute(C1_2)+S1
            Y[:,11]=np.absolute(S1_2)+C1
            Y[:,12]=np.absolute(C1_2)-S1
            
            Y[:,13]=np.absolute(C1_2)+S2
            Y[:,14]=np.absolute(S1_2)+C2
            
            Y[:,15]=C1*S2
            Y[:,16]=S1*S2
            Y[:,17]=np.absolute(C1_2)*S1
            Y[:,18]=np.absolute(S1_2)*S1
            
            Y[:,19]=S1*C2
            Y[:,20]=S1*S2
            Y[:,21]=np.absolute(S1_2)*C1
            Y[:,22]=np.absolute(S1_2)*S1
            
            Y[:,23]=S3+S2
            Y[:,24]=S3+C2
            Y[:,25]=S2*S3
            Y[:,26]=S3*C2
            
            Y[:,27]=S3+S1_2
            Y[:,28]=S3+S1_2-C2
            Y[:,29]=S3-C2
            Y[:,30]=S3-S2
                                                                                            
            m=np.expand_dims(np.min(Y,0),0)

            Y=(Y-m)/np.expand_dims(np.max(Y-m,0),0)    
        if Task=='SINE_NP_KS':
            
            if option[0]==0:
                X1=np.load(os.path.join(datafolder,'0.HDS_GS_Original.npy'),allow_pickle=True)

            if option[0]==1:
                X1=np.load(os.path.join(datafolder,'1.ASVI04_original.npy'),allow_pickle=True)

            if option[0]==2:
                X1=np.load(os.path.join(datafolder,'2.ASVI07_Original.npy'),allow_pickle=True)
            if option[0]==3:
                X1=np.load(os.path.join(datafolder,'3.Pinwheel_ASVI_y.npy'),allow_pickle=True)
            if option[0]==4:
                X1=np.load(os.path.join(datafolder,'4.Pinwheel_ASVI_x.npy'),allow_pickle=True)


            X1=X1[0:T_data,:]

            if np.shape(option)[0]==1:

                X=np.copy(X1)


            if np.shape(option)[0]>1:

                n_d=len(str(option[1]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[1])+'.'):

                            X2=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X2=X2[0:T_data,:]


                            if np.shape(option)[0]==2:

                                X=np.concatenate([X1,X2],1)


            if np.shape(option)[0]>2:

                n_d=len(str(option[2]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[2])+'.'):

                            X3=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X3=X3[0:T_data,:]

                            X=np.concatenate([X1,X2,X3],1)
            plt.plot(X[0])
            plt.show()
            sin_data2 = np.asarray(sin(800,30,1)[:800])
            
            

            data_all = SINE_TASKS()
            print(np.isnan(data_all).any())
            print(np.isinf(data_all).any())

            Y=np.zeros([T_data,len(data_all)])
            
            nans = 0
            for i in range(len(data_all)):
                #if np.isnan(data_all[i][:T_data]).any() == False:
                if 1==1:
                    Y[:,i] =data_all[i][:T_data]
                else:
                    print(i) 
                    nans+=1
                                                                                                   
            m=np.expand_dims(np.min(Y,0),0)

                  #Y=(Y-m)/np.expand_dims(np.max(Y-m,0),0)  

        if Task=='SINE_NP_KS_REDUCE':
            
            if option[0]==0:
                X1=np.load(os.path.join(datafolder,'0.HDS_GS_Original.npy'),allow_pickle=True)

            if option[0]==1:
                X1=np.load(os.path.join(datafolder,'1.ASVI04_original.npy'),allow_pickle=True)

            if option[0]==2:
                X1=np.load(os.path.join(datafolder,'2.ASVI07_Original.npy'),allow_pickle=True)
            if option[0]==3:
                X1=np.load(os.path.join(datafolder,'3.Pinwheel_ASVI_y.npy'),allow_pickle=True)
            if option[0]==4:
                X1=np.load(os.path.join(datafolder,'4.Pinwheel_ASVI_x.npy'),allow_pickle=True)


            X1=X1[0:T_data,:]

            if np.shape(option)[0]==1:

                X=np.copy(X1)


            if np.shape(option)[0]>1:

                n_d=len(str(option[1]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[1])+'.'):

                            X2=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X2=X2[0:T_data,:]


                            if np.shape(option)[0]==2:

                                X=np.concatenate([X1,X2],1)


            if np.shape(option)[0]>2:

                n_d=len(str(option[2]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[2])+'.'):

                            X3=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X3=X3[0:T_data,:]

                            X=np.concatenate([X1,X2,X3],1)
            plt.plot(X[0])
            plt.show()
            sin_data2 = np.asarray(sin(800,30,1)[:800])

            data_all = SINE_TASKS()
            print(np.isnan(data_all).any())
            print(np.isinf(data_all).any())
            Y=np.zeros([T_data,len(Task_list_inp)])
           
            nans = 0
            for i in range(len(Task_list_inp)):
                #if np.isnan(data_all[i][:T_data]).any() == False:
                if 1==1:
                    Y[:,i] =data_all[int(Task_list_inp[i])][:T_data]
                else:
                    print(i) 
                    nans+=1
                                                                                                   
            m=np.expand_dims(np.min(Y,0),0)

            #Y=(Y-m)/np.expand_dims(np.max(Y-m,0),0)         
        if Task=='SINE_NP_KS_REDUCE_FINAL':
            
            if option[0]==0:
                X1=np.load(os.path.join(datafolder,'0.HDS_GS_Original.npy'),allow_pickle=True)

            if option[0]==1:
                X1=np.load(os.path.join(datafolder,'1.ASVI04_original.npy'),allow_pickle=True)

            if option[0]==2:
                X1=np.load(os.path.join(datafolder,'2.ASVI07_Original.npy'),allow_pickle=True)
            if option[0]==3:
                X1=np.load(os.path.join(datafolder,'3.Pinwheel_ASVI_y.npy'),allow_pickle=True)
            if option[0]==4:
                X1=np.load(os.path.join(datafolder,'4.Pinwheel_ASVI_x.npy'),allow_pickle=True)


            X1=X1[0:T_data,:]

            if np.shape(option)[0]==1:

                X=np.copy(X1)


            if np.shape(option)[0]>1:

                n_d=len(str(option[1]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[1])+'.'):

                            X2=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X2=X2[0:T_data,:]


                            if np.shape(option)[0]==2:

                                X=np.concatenate([X1,X2],1)


            if np.shape(option)[0]>2:

                n_d=len(str(option[2]))

                for file in os.listdir(datafolder):

                    if '.npy' in file: 
                        if file[0:n_d+1]==(str(option[2])+'.'):

                            X3=np.load(os.path.join(datafolder,file),allow_pickle=True)
                            X3=X3[0:T_data,:]

                            X=np.concatenate([X1,X2,X3],1)
            plt.plot(X[0])
            plt.show()
            sin_data2 = np.asarray(sin(800,30,1)[:800])

            data_all = SINE_TASKS_FINAL()
            print(np.isnan(data_all).any())
            print(np.isinf(data_all).any())
            Y=np.zeros([T_data,len(data_all)])
           
            nans = 0
            for i in range(len(data_all)):
                #if np.isnan(data_all[i][:T_data]).any() == False:
                if 1==1:
                    Y[:,i] =data_all[i][:T_data]
                else:
                    print(i) 
                    nans+=1
                                                                                                   
            m=np.expand_dims(np.min(Y,0),0)

            #Y=(Y-m)/np.expand_dims(np.max(Y-m,0),0)     
        if Task=='SINE_NP_KS_REDUCE_FINAL_ALL':
            
            
            plt.plot(X[0])
            plt.show()
            sin_data2 = np.asarray(sin(800,30,1)[:800])

            data_all = SINE_TASKS_FINAL()
            print(np.isnan(data_all).any())
            print(np.isinf(data_all).any())
            Y=np.zeros([T_data,len(data_all)])
           
            nans = 0
            for i in range(len(data_all)):
                #if np.isnan(data_all[i][:T_data]).any() == False:
                if 1==1:
                    Y[:,i] =data_all[i][:T_data]
                else:
                    print(i) 
                    nans+=1
                                                                                                   
            m=np.expand_dims(np.min(Y,0),0)

            #Y=(Y-m)/np.expand_dims(np.max(Y-m,0),0)               
        if Task=='SINE':
            
            N_res=np.shape(option)[0]
            T_data=350
            self.T_data=T_data
            X=np.zeros([T_data,1])
            
            for n in range(N_res):
                
                if option[n]==0:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="ASVI04_original")
                if option[n]==1:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="ASVI07_Original")
                if option[n]==7:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="HDS_GS_Original")
                if option[n]==14:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="Pinwheel_ASVI_x")
                if option[n]==15:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="Pinwheel_ASVI_y")
                X1=df.iloc[0:T_data,3:].to_numpy()
                X=np.concatenate([X,X1],1)
            
            X=X[:,1:]
            S=df['H_app2'].to_numpy()
            S=2*(S-np.min(S))/(np.max(S-np.min(S)))-1
            S1=S[0:T_data]
            
            C1=np.zeros([T_data])
            C1[0]=np.sqrt(1-S1[0]**2)
            C1[-1]=np.sqrt(1-S1[-1]**2)
            C1[1:-1]=np.sign( (S1[2:]-S1[1:-1])/2+(S1[1:-1]-S1[0:-2])/2 )*np.sqrt(1-S1[1:-1]**2)
            
            S1_2=np.zeros([T_data])
            count=0
            S1_2=np.sqrt((1-C1)/2)
            Sg=np.zeros([T_data])
            Sg[1:-1]=np.sign( (S1_2[2:]-S1_2[1:-1])/2+(S1_2[1:-1]-S1_2[0:-2])/2 )
            Sg[0]=1
            Sg[-1]=1
            P=np.arange(0,T_data)
            P=np.reshape(P,[int(T_data/2),2])
            P=np.reshape(np.delete(P,np.arange(0,int(np.shape(P)[0]/2))*2,0),[-1])

            for t in range(1,T_data):

                if Sg[t]*Sg[t-1]<0:
                    count=count+1

                if np.any(P==count):

                    S1_2[t]=-np.sqrt((1-C1[t])/2)

                else:

                    S1_2[t]=np.sqrt((1-C1[t])/2)
            
            C1_2=np.zeros([T_data])
            C1_2[0]=np.sqrt(1-S1_2[0]**2)
            C1_2[-1]=np.sqrt(1-S1_2[-1]**2)
            C1_2[1:-1]=np.sign( (S1_2[2:]-S1_2[1:-1])/2+(S1_2[1:-1]-S1_2[0:-2])/2 )*np.sqrt(1-S1_2[1:-1]**2)

            S1_3=np.sqrt((1-C1_2)/2)
            
            C1_3=np.zeros([T_data])
            C1_3[0]=np.sqrt(1-S1_3[0]**2)
            C1_3[-1]=np.sqrt(1-S1_3[-1]**2)
            C1_3[1:-1]=np.sign( (S1_3[2:]-S1_3[1:-1])/2+(S1_3[1:-1]-S1_3[0:-2])/2 )*np.sqrt(1-S1_3[1:-1]**2)
            
            C2=C1**2-S1**2
            S2=2*S1*C1
            S3=3*S1-4*S1**3
            C3=4*C1**3-3*C1
            
            Y=np.zeros([T_data,31])
            
            Y[:,0]=S1
            Y[:,1]=S2
            Y[:,2]=S3
            Y[:,3]=C1
            Y[:,4]=C2
            Y[:,5]=np.absolute(S1_2)
            Y[:,6]=np.absolute(C1_2)
            
            Y[:,7]=S1+C2
            Y[:,8]=C1+S2
            Y[:,9]=C1-S2
            
            Y[:,10]=np.absolute(C1_2)+S1
            Y[:,11]=np.absolute(S1_2)+C1
            Y[:,12]=np.absolute(C1_2)-S1
            
            Y[:,13]=np.absolute(C1_2)+S2
            Y[:,14]=np.absolute(S1_2)+C2
            
            Y[:,15]=C1*S2
            Y[:,16]=S1*S2
            Y[:,17]=np.absolute(C1_2)*S1
            Y[:,18]=np.absolute(S1_2)*S1
            
            Y[:,19]=S1*C2
            Y[:,20]=S1*S2
            Y[:,21]=np.absolute(S1_2)*C1
            Y[:,22]=np.absolute(S1_2)*S1
            
            Y[:,23]=S3+S2
            Y[:,24]=S3+C2
            Y[:,25]=S2*S3
            Y[:,26]=S3*C2
            
            Y[:,27]=S3+S1_2
            Y[:,28]=S3+S1_2-C2
            Y[:,29]=S3-C2
            Y[:,30]=S3-S2
                                                                                            
            m=np.expand_dims(np.min(Y,0),0)

            Y=(Y-m)/np.expand_dims(np.max(Y-m,0),0)
        
        if Task=='SINE_KS_REDUCE':
            
            N_res=np.shape(option)[0]
            T_data=350
            self.T_data=T_data
            X=np.zeros([T_data,1])
            
            for n in range(N_res):
                
                if option[n]==0:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="ASVI04_original")
                if option[n]==1:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="ASVI07_Original")
                if option[n]==7:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="HDS_GS_Original")
                if option[n]==14:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="Pinwheel_ASVI_x")
                if option[n]==15:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="Pinwheel_ASVI_y")
                X1=df.iloc[0:T_data,3:].to_numpy()
                X=np.concatenate([X,X1],1)
            
            X=X[:,1:]
            S=df['H_app2'].to_numpy()
            S=2*(S-np.min(S))/(np.max(S-np.min(S)))-1
            S1=S[0:T_data]
            
            C1=np.zeros([T_data])
            C1[0]=np.sqrt(1-S1[0]**2)
            C1[-1]=np.sqrt(1-S1[-1]**2)
            C1[1:-1]=np.sign( (S1[2:]-S1[1:-1])/2+(S1[1:-1]-S1[0:-2])/2 )*np.sqrt(1-S1[1:-1]**2)
            
            S1_2=np.zeros([T_data])
            count=0
            S1_2=np.sqrt((1-C1)/2)
            Sg=np.zeros([T_data])
            Sg[1:-1]=np.sign( (S1_2[2:]-S1_2[1:-1])/2+(S1_2[1:-1]-S1_2[0:-2])/2 )
            Sg[0]=1
            Sg[-1]=1
            P=np.arange(0,T_data)
            P=np.reshape(P,[int(T_data/2),2])
            P=np.reshape(np.delete(P,np.arange(0,int(np.shape(P)[0]/2))*2,0),[-1])

            for t in range(1,T_data):

                if Sg[t]*Sg[t-1]<0:
                    count=count+1

                if np.any(P==count):

                    S1_2[t]=-np.sqrt((1-C1[t])/2)

                else:

                    S1_2[t]=np.sqrt((1-C1[t])/2)
            
            C1_2=np.zeros([T_data])
            C1_2[0]=np.sqrt(1-S1_2[0]**2)
            C1_2[-1]=np.sqrt(1-S1_2[-1]**2)
            C1_2[1:-1]=np.sign( (S1_2[2:]-S1_2[1:-1])/2+(S1_2[1:-1]-S1_2[0:-2])/2 )*np.sqrt(1-S1_2[1:-1]**2)

            S1_3=np.sqrt((1-C1_2)/2)
            
            C1_3=np.zeros([T_data])
            C1_3[0]=np.sqrt(1-S1_3[0]**2)
            C1_3[-1]=np.sqrt(1-S1_3[-1]**2)
            C1_3[1:-1]=np.sign( (S1_3[2:]-S1_3[1:-1])/2+(S1_3[1:-1]-S1_3[0:-2])/2 )*np.sqrt(1-S1_3[1:-1]**2)
            
            C2=C1**2-S1**2
            S2=2*S1*C1
            S3=3*S1-4*S1**3
            C3=4*C1**3-3*C1

            data_all = SINE_TASKS()
            print(np.isnan(data_all).any())
            print(np.isinf(data_all).any())
            Y=np.zeros([T_data,len(Task_list_inp)])
            '''
            Y[:,0]=S1
            Y[:,1]=S2
            Y[:,2]=S3
            Y[:,3]=C1
            Y[:,4]=C2
            Y[:,5]=np.absolute(S1_2)
            Y[:,6]=np.absolute(C1_2)
            
            Y[:,7]=S1+C2
            Y[:,8]=C1+S2
            Y[:,9]=C1-S2
            
            Y[:,10]=np.absolute(C1_2)+S1
            Y[:,11]=np.absolute(S1_2)+C1
            Y[:,12]=np.absolute(C1_2)-S1
            
            Y[:,13]=np.absolute(C1_2)+S2
            Y[:,14]=np.absolute(S1_2)+C2
            
            Y[:,15]=C1*S2
            Y[:,16]=S1*S2
            Y[:,17]=np.absolute(C1_2)*S1
            Y[:,18]=np.absolute(S1_2)*S1
            
            Y[:,19]=S1*C2
            Y[:,20]=S1*S2
            Y[:,21]=np.absolute(S1_2)*C1
            Y[:,22]=np.absolute(S1_2)*S1
            
            Y[:,23]=S3+S2
            Y[:,24]=S3+C2
            Y[:,25]=S2*S3
            Y[:,26]=S3*C2
            
            Y[:,27]=S3+S1_2
            Y[:,28]=S3+S1_2-C2
            Y[:,29]=S3-C2
            Y[:,30]=S3-S2
            '''
            nans = 0
            for i in range(len(Task_list_inp)):
                #if np.isnan(data_all[i][:T_data]).any() == False:
                if 1==1:
                    Y[:,i] =data_all[int(Task_list_inp[i])][:T_data]
                else:
                    print(i) 
                    nans+=1
                                                                                                   
            m=np.expand_dims(np.min(Y,0),0)

            #Y=(Y-m)/np.expand_dims(np.max(Y-m,0),0) 

        if Task=='SINE_KS':
            
            N_res=np.shape(option)[0]
            T_data=350
            self.T_data=T_data
            X=np.zeros([T_data,1])
            
            for n in range(N_res):
                
                if option[n]==0:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="ASVI04_original")
                if option[n]==1:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="ASVI07_Original")
                if option[n]==7:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="HDS_GS_Original")
                if option[n]==14:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="Pinwheel_ASVI_x")
                if option[n]==15:
                    df = pd.read_excel('Sin_originals_paper.xlsx',sheet_name="Pinwheel_ASVI_y")
                X1=df.iloc[0:T_data,3:].to_numpy()
                X=np.concatenate([X,X1],1)
            
            X=X[:,1:]
            S=df['H_app2'].to_numpy()
            S=2*(S-np.min(S))/(np.max(S-np.min(S)))-1
            S1=S[0:T_data]
            
            C1=np.zeros([T_data])
            C1[0]=np.sqrt(1-S1[0]**2)
            C1[-1]=np.sqrt(1-S1[-1]**2)
            C1[1:-1]=np.sign( (S1[2:]-S1[1:-1])/2+(S1[1:-1]-S1[0:-2])/2 )*np.sqrt(1-S1[1:-1]**2)
            
            S1_2=np.zeros([T_data])
            count=0
            S1_2=np.sqrt((1-C1)/2)
            Sg=np.zeros([T_data])
            Sg[1:-1]=np.sign( (S1_2[2:]-S1_2[1:-1])/2+(S1_2[1:-1]-S1_2[0:-2])/2 )
            Sg[0]=1
            Sg[-1]=1
            P=np.arange(0,T_data)
            P=np.reshape(P,[int(T_data/2),2])
            P=np.reshape(np.delete(P,np.arange(0,int(np.shape(P)[0]/2))*2,0),[-1])

            for t in range(1,T_data):

                if Sg[t]*Sg[t-1]<0:
                    count=count+1

                if np.any(P==count):

                    S1_2[t]=-np.sqrt((1-C1[t])/2)

                else:

                    S1_2[t]=np.sqrt((1-C1[t])/2)
            
            C1_2=np.zeros([T_data])
            C1_2[0]=np.sqrt(1-S1_2[0]**2)
            C1_2[-1]=np.sqrt(1-S1_2[-1]**2)
            C1_2[1:-1]=np.sign( (S1_2[2:]-S1_2[1:-1])/2+(S1_2[1:-1]-S1_2[0:-2])/2 )*np.sqrt(1-S1_2[1:-1]**2)

            S1_3=np.sqrt((1-C1_2)/2)
            
            C1_3=np.zeros([T_data])
            C1_3[0]=np.sqrt(1-S1_3[0]**2)
            C1_3[-1]=np.sqrt(1-S1_3[-1]**2)
            C1_3[1:-1]=np.sign( (S1_3[2:]-S1_3[1:-1])/2+(S1_3[1:-1]-S1_3[0:-2])/2 )*np.sqrt(1-S1_3[1:-1]**2)
            
            C2=C1**2-S1**2
            S2=2*S1*C1
            S3=3*S1-4*S1**3
            C3=4*C1**3-3*C1

            data_all = SINE_TASKS()
            print(np.isnan(data_all).any())
            print(np.isinf(data_all).any())

            Y=np.zeros([T_data,len(data_all)])
            '''
            Y[:,0]=S1
            Y[:,1]=S2
            Y[:,2]=S3
            Y[:,3]=C1
            Y[:,4]=C2
            Y[:,5]=np.absolute(S1_2)
            Y[:,6]=np.absolute(C1_2)
            
            Y[:,7]=S1+C2
            Y[:,8]=C1+S2
            Y[:,9]=C1-S2
            
            Y[:,10]=np.absolute(C1_2)+S1
            Y[:,11]=np.absolute(S1_2)+C1
            Y[:,12]=np.absolute(C1_2)-S1
            
            Y[:,13]=np.absolute(C1_2)+S2
            Y[:,14]=np.absolute(S1_2)+C2
            
            Y[:,15]=C1*S2
            Y[:,16]=S1*S2
            Y[:,17]=np.absolute(C1_2)*S1
            Y[:,18]=np.absolute(S1_2)*S1
            
            Y[:,19]=S1*C2
            Y[:,20]=S1*S2
            Y[:,21]=np.absolute(S1_2)*C1
            Y[:,22]=np.absolute(S1_2)*S1
            
            Y[:,23]=S3+S2
            Y[:,24]=S3+C2
            Y[:,25]=S2*S3
            Y[:,26]=S3*C2
            
            Y[:,27]=S3+S1_2
            Y[:,28]=S3+S1_2-C2
            Y[:,29]=S3-C2
            Y[:,30]=S3-S2
            '''
            nans = 0
            for i in range(len(data_all)):
                #if np.isnan(data_all[i][:T_data]).any() == False:
                if 1==1:
                    Y[:,i] =data_all[i][:T_data]
                else:
                    print(i) 
                    nans+=1
                                                                                                   
            m=np.expand_dims(np.min(Y,0),0)

            #Y=(Y-m)/np.expand_dims(np.max(Y-m,0),0)     
        
        self.Task=Task
                        
        self.X_original=X
        self.Y_original=Y
        
        self.X=X
        self.Y=Y
        
        self.X_tr=[]
        self.X_val=[]
        self.X_te=[]
        
        self.X_tr_M=[]
        self.X_val_M=[]
        self.X_te_M=[]
        
        self.Y_tr=[]
        self.Y_val=[]
        self.Y_te=[]
        
        self.it_ind=[]
        
        self.Corr=[]
        
        self.Cov=[]
        self.w=[]
        self.v=[]
        
        self.p_te=p_te
        
        self.index_tr=[]
        self.index_val=[]
        self.index_te=[]
        
        self.Z_val=[]
        self.Z_te=[]
        
    def CrossVal(self,te_ind,val_ind):
    
        N_data=np.shape(self.X)[0]
        
        N_trans=20
        
        N_te=int(np.floor((N_data-N_trans)*self.p_te))
        
        N_val=int(np.floor((N_data-N_trans-N_te)*0.1))
        
        index=np.arange(N_trans,N_data)
        
        index_te=np.copy(index[te_ind*N_te:(te_ind+1)*N_te])
        index_noT=np.delete(index,index_te-N_trans)
        
        index_val=np.copy(index_noT[val_ind*N_val:(val_ind+1)*N_val])
        
        index_tr=np.delete(index,np.concatenate([index_te-N_trans,index_val-N_trans],0))
        
        
        X_tr=np.copy(self.X[index_tr,:])
        
        X_M=np.max(np.abs(X_tr),0)
        
        X_tr=X_tr/np.tile(np.expand_dims(X_M,0),[np.shape(X_tr)[0],1])
        X_val=np.copy(self.X[index_val,:])/np.tile(np.expand_dims(X_M,0),[N_val,1])
        X_te=np.copy(self.X[index_te,:])/np.tile(np.expand_dims(X_M,0),[N_te,1])
        
        Y_tr=np.copy(self.Y[index_tr,:])
        Y_val=np.copy(self.Y[index_val,:])
        Y_te=np.copy(self.Y[index_te,:])
        
        self.X_tr_M=np.copy(X_tr)
        self.X_val_M=np.copy(X_val)
        self.X_te_M=np.copy(X_te)
        
        self.Y_tr_M=np.copy(Y_tr)
        self.Y_val_M=np.copy(Y_val)
        self.Y_te_M=np.copy(Y_te)
        
        self.X_tr=torch.from_numpy(X_tr).float()
        self.X_te=torch.from_numpy(X_te).float()
        self.X_val=torch.from_numpy(X_val).float()
        
        self.Y_tr=torch.from_numpy(Y_tr).float()
        self.Y_te=torch.from_numpy(Y_te).float()
        self.Y_val=torch.from_numpy(Y_val).float()
        
        self.index_val=index_val
        self.index_te=index_te
        
        
    def CORR(self):
        
        Corr=np.corrcoef(np.transpose(self.X_tr_M))
        
        self.Corr=Corr
        
        
    def CORR_Analysis(self,th,removal):
        
        
        N=np.shape(self.Corr)[0]
        
        N_red=np.abs(self.Corr)>th
        
        RM=[]
        
        for i in range(N):
            
            if any(N_red[i,:]==1):
                
                
                ind=np.where(N_red[i,:]==1)[0]
                
                if any(ind>i):
                    
                    RM.append(i)
        
        RM=np.array(RM)
        
        
        if removal==True and RM!=[]:
                
            self.X_tr_M=np.delete(self.X_tr,RM,1)    
            self.X_val_M=np.delete(self.X_val,RM,1)
            self.X_te_M=np.delete(self.X_te,RM,1)
            
        else:
            
            self.X_tr_M=self.X_tr    
            self.X_val_M=self.X_val
            self.X_te_M=self.X_te
            
        
        return RM
    
    def Benchmark_Correlation(self,ths):
        
        
        N_fit=np.shape(ths)[0]
        

        alphas=np.array([1e-4,1e-3,1e-2,5*1e-2,1*1e-1])        
        N_alpha=np.shape(alphas)[0]
        
        N_out=np.shape(self.Y_tr)[1]
        
        N_split=10
        
        N=np.shape(self.X_tr_M)[1]
        
        N_split_te=int(1/self.p_te)
        
        RM=np.zeros([N_split_te,N])
        
        Best_hyper=np.zeros([N_split_te,3])
        
        
        MSE_Te=np.zeros([N_split_te,N_out])
        MSE_Val=np.zeros([N_split_te])
        MSE_Tr=np.zeros([N_split_te])
        
        Z_val=np.zeros([N_split_te,self.T_data,N_out,2])
        Z_te=np.zeros([self.T_data,N_out,2])
        
        for l in range(N_split_te):
            
            mse_Val=np.zeros([N_split,np.shape(ths)[0],N_alpha,N_out])
            mse_Tr=np.zeros([N_split,np.shape(ths)[0],N_alpha,N_out])
            O_val=np.zeros([N_fit,N_alpha,self.T_data,N_out,2])
        
            for j in range(N_split):
                
                self.CrossVal(l,j)
                self.CORR()
                
                
                for i in range(N_fit):
                                    
                    _=self.CORR_Analysis(ths[i],True)
                                    
                    
                    for k in range(N_alpha):
                
                        model=Ridge(alpha = alphas[k],fit_intercept = False,copy_X = True)
                        reg = model.fit(self.X_tr_M, self.Y_tr_M)
            
                        output_val = reg.predict(self.X_val_M)
                        output_train = reg.predict(self.X_tr_M)
                        O_val[i,k,self.index_val,:,0]=np.copy(output_val)
                        O_val[i,k,self.index_val,:,1]=np.copy(self.Y_val_M)
                        
                        mse_Tr[j,i,k,:] = np.mean((self.Y_tr_M-output_train)**2,0)
                        mse_Val[j,i,k,:] = np.mean((self.Y_val_M-output_val)**2,0)
                        
                        
            M_Tr=np.mean(np.mean(mse_Tr,0),2) 
            M_Val=np.mean(np.mean(mse_Val,0),2) 
            
            ind=np.argwhere(M_Val==np.min(M_Val))[0]
            
            MSE_Val[l]=np.copy(M_Val[ind[0],ind[1]])
            MSE_Tr[l]=np.copy(M_Tr[ind[0],ind[1]])
    
            Z_val[l,:,:,0]=np.copy(O_val[ind[0],ind[1],:,:,0])
            Z_val[l,:,:,1]=np.copy(O_val[ind[0],ind[1],:,:,1])
            
            self.X_tr_M=np.concatenate([self.X_tr,self.X_val],0)
            
            self.CORR()
            
            EL_removed=self.CORR_Analysis(ths[ind[0]],True)
            
            if EL_removed!=[]:
                RM[l,EL_removed]=1
            
                
            Best_hyper[l,0]=ths[ind[0]]
            Best_hyper[l,1]=alphas[ind[1]]
            Best_hyper[l,2]=np.sum(RM[l,:]==0)
            
            self.X_tr_M=np.concatenate([self.X_tr_M,self.X_val_M],0)
            self.Y_tr_M=np.concatenate([self.Y_tr_M,self.Y_val_M],0)
            
            model=Ridge(alpha = alphas[ind[1]],fit_intercept = False,copy_X = True)
            reg = model.fit(self.X_tr_M, self.Y_tr_M)
            
            output_test = reg.predict(self.X_te_M)
            
            MSE_Te[l,:] = np.mean((self.Y_te_M-output_test)**2,0)
                
            
            Z_te[self.index_te,:,0]=np.copy(output_test)
            Z_te[self.index_te,:,1]=np.copy(self.Y_te_M)
            
        
        print('Performance: TE, ', np.mean(MSE_Te), 'VAL ', np.mean(MSE_Val), 'TR ', np.mean(MSE_Tr))

        
        return MSE_Tr, MSE_Val, MSE_Te, RM, Best_hyper, Z_val, Z_te
    
    def Measure_MC(self,ths):
        
        MSE_Tr, MSE_Val, MSE_Te, RM, Best_hyper, Z_val, Z_te=self.Benchmark_Correlation(ths)
        
        MC=np.zeros([np.shape(Z_te)[1]])
        
        for k in range(np.shape(Z_te)[1]):
            
            ind_te=np.where(Z_te[:,k,0]!=0)
            u_test=np.copy(Z_te[ind_te,k,0])
            u_pred=np.copy(Z_te[ind_te,k,1])
            
            MC[k]=( np.mean(u_test*u_pred) - np.mean(u_test)*np.mean(u_pred) )**2 / (np.var(u_test) * np.var(u_pred))
        
        
        print('MC: ', np.sum(MC))
        
        return MSE_Tr, MSE_Val, MSE_Te, RM, Best_hyper, Z_val, Z_te, MC
    
    
    def Measure_NL(self):
        
        k_min=5
        N_fit=self.k_max-k_min
        Ks=np.arange(k_min,self.k_max+1)

        alphas=np.array([1e-4,1e-3,1e-2,5*1e-2,1*1e-1])        
        N_alpha=np.shape(alphas)[0]
        
        N_out=np.shape(self.Y_tr)[1]
        
        N_split=10
        
        N_split_te=int(1/self.p_te)
        
        Best_hyper=np.zeros([N_split_te,3])
        
        MSE_Te=np.zeros([N_split_te,N_out])
        MSE_Val=np.zeros([N_split_te])
        MSE_Tr=np.zeros([N_split_te])
        
        Z_val=np.zeros([N_split_te,self.T_data,N_out,2])
        Z_te=np.zeros([self.T_data,N_out,2])
        
        for l in range(N_split_te):
            
            mse_Val=np.zeros([N_split,N_fit,N_alpha,N_out])
            mse_Tr=np.zeros([N_split,N_fit,N_alpha,N_out])
            O_val=np.zeros([N_fit,N_alpha,self.T_data,N_out,2])
        
            for j in range(N_split):
                
                self.CrossVal(l,j)
                
                for i in range(N_fit):
                                    
                    self.X_tr_M=np.copy(self.X_tr[:,0:Ks[i]])    
                    self.X_val_M=np.copy(self.X_val[:,0:Ks[i]])
                    self.X_te_M=np.copy(self.X_te[:,0:Ks[i]])
                                    
                    for k in range(N_alpha):
                
                        model=Ridge(alpha = alphas[k],fit_intercept = False,copy_X = True)
                        reg = model.fit(self.X_tr_M, self.Y_tr_M)
            
                        output_val = reg.predict(self.X_val_M)
                        output_train = reg.predict(self.X_tr_M)
                        O_val[i,k,self.index_val,:,0]=np.copy(output_val)
                        O_val[i,k,self.index_val,:,1]=np.copy(self.Y_val_M)
                        
                        mse_Tr[j,i,k,:] = np.mean((self.Y_tr_M-output_train)**2,0)
                        mse_Val[j,i,k,:] = np.mean((self.Y_val_M-output_val)**2,0)
                        
                  
            M_Tr=np.mean(np.mean(mse_Tr,0),2) 
            M_Val=np.mean(np.mean(mse_Val,0),2) 
            
            
            ind=np.argwhere(M_Val==np.min(M_Val))[0]
            
            MSE_Val[l]=np.copy(M_Val[ind[0],ind[1]])
            MSE_Tr[l]=np.copy(M_Tr[ind[0],ind[1]])
    
            Z_val[l,:,:,0]=np.copy(O_val[ind[0],ind[1],:,:,0])
            Z_val[l,:,:,1]=np.copy(O_val[ind[0],ind[1],:,:,1])
            
            self.X_tr_M=np.concatenate([self.X_tr,self.X_val],0)
            self.X_tr_M=self.X_tr_M[:,0:Ks[ind[0]]]
            self.X_te_M=self.X_te_M[:,0:Ks[ind[0]]]
            
            self.Y_tr_M=np.concatenate([self.Y_tr_M,self.Y_val_M],0)
            
            
            Best_hyper[l,0]=Ks[ind[0]]
            Best_hyper[l,1]=alphas[ind[1]]

            model=Ridge(alpha = alphas[ind[1]],fit_intercept = False,copy_X = True)
            reg = model.fit(self.X_tr_M, self.Y_tr_M)
            
            output_test = reg.predict(self.X_te_M)
            
            MSE_Te[l,:] = np.mean((self.Y_te_M-output_test)**2,0)
                
            
            Z_te[self.index_te,:,0]=np.copy(output_test)
            Z_te[self.index_te,:,1]=np.copy(self.Y_te_M)
            
        
        print('Performance: TE, ', np.mean(MSE_Te), 'VAL ', np.mean(MSE_Val), 'TR ', np.mean(MSE_Tr))
        
        MC=np.zeros([np.shape(Z_te)[1]])
        
        for k in range(np.shape(Z_te)[1]):
            
            ind_te=np.where(Z_te[:,k,0]!=0)
            u_test=np.copy(Z_te[ind_te,k,0])
            u_pred=np.copy(Z_te[ind_te,k,1])
            
            MC[k]=( np.mean(u_test*u_pred) - np.mean(u_test)*np.mean(u_pred) )**2 / (np.var(u_test) * np.var(u_pred))
        
        
        NL=1-np.mean(MC)
        
        print('NL: ', NL)
        return MSE_Tr, MSE_Val, MSE_Te, Best_hyper, Z_val, Z_te, MC
        
    
    
    def Evolutionary(self, Feats, p_mutation, N_iteration, alphas):
        
        Parents_in=False
        
        N_children=200
        N_split=10
        N_best=10
        
        N_split_te=int(1/self.p_te)
        
        N_out=np.shape(self.Y)[1]
        N_feat=np.shape(self.X)[1]

        MSE1=np.zeros([N_split_te,N_iteration,N_best,N_out])
        MSE2=np.zeros([N_split_te,N_iteration,N_best,N_out])
        
        Z_val=np.zeros([N_split_te,self.T_data,N_out,3])
        Z_te=np.zeros([self.T_data,N_out,2])
        
        MSE=[]
        MSE_bench=[]
        
        for split in range(N_split_te):
            
            Mask=np.copy(Feats[split,:])
            
            alpha=np.copy(alphas[split])
            
            model=Ridge(alpha = alpha,fit_intercept = False,copy_X = True)
            best_ch=[]

            self.CrossVal(split,9)

            N_te=np.shape(self.X_te)[0]
            N_te=int(N_te/2)
            
            index_te1=self.index_te[0:N_te]
            index_te2=self.index_te[N_te:]
            
            X_te1=np.copy(self.X_te[0:N_te,:])
            Y_te1=np.copy(self.Y_te[0:N_te,:])

            X_te2=np.copy(self.X_te[N_te:,:])
            Y_te2=np.copy(self.Y_te[N_te:,:])

            MSE_te1=np.zeros([N_iteration,N_best])
            MSE_te2=np.zeros([N_iteration,N_best])

            RM=np.where(Mask==1)

            X_te1_M=np.delete(X_te1,RM,1)
            X_te2_M=np.delete(X_te2,RM,1)

            self.X_tr_M=np.delete(np.concatenate([self.X_tr,self.X_val],0),RM,1)
            self.Y_tr_M=np.concatenate([self.Y_tr,self.Y_val],0)

            reg=model.fit(self.X_tr_M, self.Y_tr_M)

            MSE_bench.append(np.mean((Y_te1-reg.predict(X_te1_M))**2))
            MSE_bench.append(np.mean((Y_te2-reg.predict(X_te2_M))**2))
            
            print('Split Number ', split)
            print('Before Evolution: ', MSE_bench[-2],MSE_bench[-1])
            
            if Parents_in:

                N_loop=N_children+N_best

            if not(Parents_in):
                
                N_loop=N_children
            
            
            O_val=np.zeros([N_iteration,N_loop,self.T_data,N_out,2])
            O_te=np.zeros([N_iteration,N_best,self.T_data,N_out,2])
            
            for n in range(N_iteration):

                if n==0:    

                    if Parents_in:

                        Masks=np.tile(np.expand_dims(Mask,1),[1,N_children+N_best])

                    if not(Parents_in):

                        Masks=np.tile(np.expand_dims(Mask,1),[1,N_children])


                else:

                    Parents=np.copy(Masks[:,best_ch])

                    ind_mix=np.random.permutation(N_best)
                    Masks=np.zeros([N_feat,N_children])
                    N_mix=int(2*N_children/N_best)

                    for l in range(int(N_best/2)):

                        cross=np.random.rand(N_feat,N_mix)<0.5

                        Masks[:,N_mix*l:N_mix*(l+1)]=np.tile(np.expand_dims(Parents[:,ind_mix[2*l]],1),[1,N_mix])*cross+np.tile(np.expand_dims(Parents[:,ind_mix[2*l+1]],1),[1,N_mix])*(1-cross)


                    if Parents_in:

                        Masks=np.concatenate([Masks,Parents],1)


                Mutations=np.random.rand(N_feat,N_children)<p_mutation

                if Parents_in:

                    Masks[:,0:N_children]=Masks[:,0:N_children]*(1-Mutations)+(1-Masks[:,0:N_children])*Mutations

                if not(Parents_in):
                    
                    Masks=Masks*(1-Mutations)+(1-Masks)*Mutations

                MSE_val=np.zeros([N_loop,N_split])
                                
                for i in range(N_loop):

                    RM=np.where(Masks[:,i]==1)


                    for j in range(N_split):

                        self.CrossVal(split,j)

                        self.X_tr_M=np.delete(self.X_tr,RM,1)    
                        self.X_val_M=np.delete(self.X_val,RM,1)

                        reg=model.fit(self.X_tr_M, self.Y_tr_M)

                        MSE_val[i,j]=np.mean((self.Y_val_M-reg.predict(self.X_val_M))**2)
                        
                        O_val[n,i,self.index_val,:,0]= reg.predict(self.X_val_M)
                        O_val[n,i,self.index_val,:,1]= np.copy(self.Y_val_M)
                                                
                        
                best_ch=np.argsort(np.mean(MSE_val,1))[0:N_best]
                
                O_val[:,0:N_best,:,:]=O_val[:,best_ch,:,:,:]
                
                for i in range(N_best):

                    RM=np.where(Masks[:,best_ch[i]]==1)

                    self.X_tr_M=np.delete(np.concatenate([self.X_tr,self.X_val],0),RM,1)
                    self.Y_tr_M=np.concatenate([self.Y_tr,self.Y_val],0)

                    reg=model.fit(self.X_tr_M, self.Y_tr_M)

                    X_te1_M=np.delete(X_te1,RM,1)
                    X_te2_M=np.delete(X_te2,RM,1)
                    
                    MSE1[split,n,i,:]=np.mean((Y_te1-reg.predict(X_te1_M))**2,0)
                    MSE2[split,n,i,:]=np.mean((Y_te2-reg.predict(X_te2_M))**2,0)
                    
                    MSE_te1[n,i]=np.mean(MSE1[split,n,i,:])
                    MSE_te2[n,i]=np.mean(MSE2[split,n,i,:])
                    
                    O_te[n,i,index_te1,:,0]= reg.predict(X_te1_M)
                    O_te[n,i,index_te2,:,0]= reg.predict(X_te2_M)
                    O_te[n,i,index_te1,:,1]= np.copy(Y_te1)
                    O_te[n,i,index_te2,:,1]= np.copy(Y_te2)
                    
            O_val=O_val[:,0:N_best,:,:]
            
            ind1=np.argwhere(MSE_te1==np.min(MSE_te1))[0]
            ind2=np.argwhere(MSE_te2==np.min(MSE_te2))[0]
            
            Z_val[split,:,:,0]=np.copy(O_val[ind2[0],ind2[1],:,:,0])
            Z_val[split,:,:,1]=np.copy(O_val[ind1[0],ind1[1],:,:,0])
            Z_val[split,:,:,2]=np.copy(O_val[ind1[0],ind1[1],:,:,1])
            
            Z_te[index_te1,:,0]=O_te[ind2[0],ind2[1],index_te1,:,0]
            Z_te[index_te2,:,0]=O_te[ind1[0],ind1[1],index_te2,:,0]
            Z_te[self.index_te,:,1]=O_te[ind1[0],ind1[1],self.index_te,:,1]
            
            
            MSE.append(MSE1[split,ind2[0],ind2[1],:])
            MSE.append(MSE2[split,ind1[0],ind1[1],:])
            
            print('After Evolution: ', MSE_te1[ind2[0],ind2[1]],MSE_te2[ind1[0],ind1[1]], ind1[0], ind2[0])
            print('Minima (Overfitting): ', np.min(MSE_te1),np.min(MSE_te2))
        
        MSE=np.array(MSE)
        MSE_bench=np.array(MSE_bench)
        
        print('Overall, Before and After: ', np.mean(MSE_bench), np.mean(MSE))
            
        return MSE, MSE_bench, Z_val, Z_te

        
class Regression_ReadOuts:
    
    def __init__(self,N,N_class,batch_size):
        
        
        self.N=N
        
        self.N_class=N_class
                
        self.batch_size=batch_size
                
        self.Ws=[]
        
        self.theta_g=[]
        
        self.theta_i=[]
        
        self.loss=[]
        self.opt=[]
        self.opt_theta=[]
        
        self.N_copies=[]
        
    
    def Initialise_Online(self,scan,N_copies,alpha_size):
        
        
        self.N_copies=N_copies
        if scan==False:
            
            self.N_copies=1
            alpha_sizes=[alpha_size]
                               
        
        if scan==True:
            
            alpha_sizes=0.01*2**(-np.linspace(0,7,self.N_copies))

        self.loss = nn.MSELoss()
        
        for i in range(self.N_copies):
            
            
            self.Ws.append(nn.Parameter( (2*torch.rand([self.N,self.N_class])-1)/(self.N/10)))
            self.opt.append(optim.Adam([{'params': self.Ws, 'lr':alpha_sizes[i] }]))
        
        
    
    def Online_Step(self,state,y_true):
        
       
        y=[]
        error=[]
        
        loss = nn.MSELoss()
                
        for i in range(self.N_copies):
            
            
            y.append(torch.matmul(state,self.Ws[i]) )
            
            error.append(loss(y[i],y_true))
            error[i].backward()

        
            self.opt[i].step()
            self.opt[i].zero_grad()
            
            
                
        return y, error
    
    
    def Online_Evaluate(self,state,y_true):
    
        y=[]
        error=[]            
        
        loss = nn.MSELoss()
                
        for i in range(self.N_copies):
            
            
            y.append(torch.matmul(state,self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            
            
        return y, error
    


    def Initialise_SpaRCe(self,X_tr,alpha_size):
        
        
        Pns=[0,10,20,30,40,50,60,70,80,90]
        
        self.N_copies=np.shape(Pns)[0]
        
        self.loss = nn.MSELoss()
        
        
        for i in range(self.N_copies):
            
            
            theta_g_start=np.percentile(np.abs(X_tr),Pns[i],0)
            
            self.theta_g.append(torch.from_numpy(theta_g_start).float())
            
            
            self.theta_i.append(nn.Parameter(torch.zeros([self.N])))
            
            self.Ws.append(nn.Parameter( (2*torch.rand([self.N,self.N_class])-1)/(self.N/10) ))
            
            self.opt.append(optim.Adam([{'params': self.Ws, 'lr':alpha_size },{'params': self.theta_i, 'lr':alpha_size/10 }]))
            
            
    
    def SpaRCe_Step(self,state,y_true):
        
        
        state_sparse=[]
        y=[]
        error=[]
        
        loss = nn.MSELoss()

        for i in range(self.N_copies):
            
            
            state_sparse.append(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i]))     
    
            y.append(torch.matmul(state_sparse[i],self.Ws[i]))
                
            error.append(loss(y[i],y_true))
            
            error[i].backward()

            self.opt[i].step()
            self.opt[i].zero_grad()
            
    
            
        return y, error, state_sparse
    
    def SpaRCe_Evaluate(self,state,y_true):
    
        state_sparse=[]
        y=[]
        error=[]            
        sparsity=[]
        
        loss = nn.MSELoss()
        
        N_cl=torch.sum(state!=0)
        
        for i in range(self.N_copies):
            
            
            state_sparse.append(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i]))     
    
            y.append(torch.matmul(state_sparse[i],self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            
            sparsity.append(torch.sum(state_sparse[i]!=0)/N_cl)
            
        return y, error, sparsity, state_sparse
            
    
    def Ridge_Regression(self,X_tr,Y_tr,X_val,Y_val,X_te,Y_te):
    
        
        gammas=[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256]
        
        
        self.N_copies=np.shape(gammas)[0]
        self.Ws=[]
        y_tr=[]
        y_te=[]
        y_val=[]
        
        Err_tr=[]
        Err_te=[]
        Err_val=[]
        
        for i in range(self.N_copies):
            
            Y=torch.transpose(Y_tr,0,1)
            
            W=torch.matmul( torch.matmul(Y,X_tr), torch.inverse( torch.matmul(torch.transpose(X_tr,0,1),X_tr) + gammas[i]*torch.eye(self.N) ) )
            
            self.Ws.append(torch.transpose(W,0,1))
                    
            y_tr.append(torch.matmul(X_tr,self.Ws[i]))
            
            y_te.append(torch.matmul(X_te,self.Ws[i]))
            
            y_val.append(torch.matmul(X_val,self.Ws[i]))

            Err_tr.append(torch.mean((y_tr[i]-Y_tr)**2,0))
            Err_val.append(torch.mean((y_val[i]-Y_val)**2,0))
            Err_te.append(torch.mean((y_te[i]-Y_te)**2,0))
           
            
        return y_tr, y_val, y_te, Err_tr, Err_val, Err_te
    


class Pruning:


    def __init__(self, W_Best, theta_g_Best, theta_i_Best, N_copies):
        
        
        self.N=W_Best.size()[0]
        
        self.N_class=W_Best.size()[1]
                
        self.N_copies=N_copies
        
        self.W_Best=W_Best
        
        self.Ws=[]
        self.theta_g=[]
        self.theta_i=[]
        
        for i in range(N_copies):         
            
            self.Ws.append(W_Best)
        
            self.theta_g.append(theta_g_Best)
        
            self.theta_i.append(theta_i_Best)
        
        
    def SpaRCe_Evaluate(self,state,y_true):
    
        state_sparse=[]
        y=[]
        Acc=[]    
        error=[]            
        sparsity=[]
        
        loss = nn.BCEWithLogitsLoss()
        
        N_cl=torch.sum(state!=0)
        
        for i in range(self.N_copies):
            
            
            state_sparse.append(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i]))     
    
            y.append(torch.matmul(state_sparse[i],self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            Acc.append(torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() ))
            
            sparsity.append(torch.sum(state_sparse[i]!=0)/N_cl)
            
        return y, Acc, error, state_sparse
    
    
    
    def Online_Evaluate(self,state,y_true):
    
        y=[]
        Acc=[]    
        error=[]            
        
        loss = nn.BCEWithLogitsLoss()
                
        for i in range(self.N_copies):
            
            
            y.append(torch.matmul(state,self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            Acc.append(torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() ))
            
            
        return y, Acc, error
            
    
    
    
    def Prune(self,  X_tr, Y_tr, X_te, Y_te, SpaRCe_True, N_cuts):
        
        
        
        images=torch.clone(X_tr[:,:])
        labels=torch.clone(Y_tr[:,:])
        
        
        if SpaRCe_True:
            
            Out_tr, Acc_tr, Error_tr, S=self.SpaRCe_Evaluate(images,labels)
            
            
            Active=torch.mean((S[0]!=0).float(),0)

            th=np.linspace(0.0,1,201)
            N_cuts=np.zeros(np.shape(th)[0])
            
            for i in range(np.shape(th)[0]):
                
                N_cuts[i]=torch.sum((Active>th[i])==0)
                
                
        else:
            
            Out_tr, Acc_tr, Error_tr=self.Online_Evaluate(images,labels)
            
            print('Provide the number of nodes to be deleted')
            print(N_cuts)
            
        
        Out_Sp=[]
        Acc_Sp=[]
        Err_Sp=[]
        
        Out_Rand=[]
        Acc_Rand=[]
        Err_Rand=[]
        
        Out_W=[]
        Acc_W=[]
        Err_W=[]
                
        
        for i in range(np.shape(N_cuts)[0]):
            
            
            images=torch.clone(X_te[:,:])
            labels=torch.clone(Y_te[:,:])
            
            N_cut=np.copy(N_cuts[i])
            
            
            if SpaRCe_True:
            
                Mask=Active>th[i]
                Mask=torch.unsqueeze(Mask,1).repeat(1,self.N_class).float()
                self.Ws[0]=torch.clone(nn.Parameter(self.W_Best*Mask))
            
                
                Out_te_Sp, Acc_te_Sp, Error_te_Sp, _=self.SpaRCe_Evaluate(images,labels)
                
            
                Out_Sp.append(Out_te_Sp[0].detach())
                Acc_Sp.append(Acc_te_Sp[0].detach())
                Err_Sp.append(Error_te_Sp[0].detach())
            
            
            for j in range(self.N_copies):
                
                Mask=torch.randint(0,self.N,[int(N_cut)])
                self.Ws[j]=torch.clone(nn.Parameter(self.W_Best))
                self.Ws[j][Mask,:]=0
            
            
            if SpaRCe_True:
            
                Out_te_Rand, Acc_te_Rand, Error_te_Rand, _=self.SpaRCe_Evaluate(images,labels)
            
            else:
                
                Out_te_Rand, Acc_te_Rand, Error_te_Rand=self.Online_Evaluate(images,labels)
                
            
            Out_Rand.append(Out_te_Rand)
            Acc_Rand.append(Acc_te_Rand)
            Err_Rand.append(Error_te_Rand)
            
            
            Fisher=torch.matmul((1-torch.sigmoid(torch.transpose(Out_tr[0],0,1))),X_tr)**2
            sort, indexes=torch.sort(torch.mean(Fisher,0))
            
                        
            Mask=indexes[0:int(N_cut)]
            self.Ws[0]=torch.clone(nn.Parameter(self.W_Best))
            self.Ws[0][Mask,:]=0
            
            if SpaRCe_True:
            
                Out_te_W, Acc_te_W, Error_te_W, _=self.SpaRCe_Evaluate(images,labels)
                
            else:
            
                Out_te_W, Acc_te_W, Error_te_W=self.Online_Evaluate(images,labels)
            
            
            Out_W.append(Out_te_W[0].detach())
            Acc_W.append(Acc_te_W[0].detach())
            Err_W.append(Error_te_W[0].detach())
            
            
        if SpaRCe_True:
            
            return Out_Sp, Acc_Sp, Err_Sp, Out_Rand, Acc_Rand, Err_Rand, Out_W, Acc_W, Err_W, N_cuts
        
        
        else:
        
            return Out_Rand, Acc_Rand, Err_Rand, Out_W, Acc_W, Err_W
        
        
        
        
def TI46Performance(Out, Y, Ts):
    
    N_te=np.shape(Ts)[0]
    
    ind_start=0
    ind_end=0
    p=0

    for j in range(N_te):

        ind_end=ind_end+Ts[j]

        p=p+np.float32( np.argmax( np.sum(Out[ind_start:ind_end,:],0) )==np.argmax( Y[ind_start,:] ) )/N_te
        
        ind_start=np.copy(ind_end)  
        
    return p
    
            
    
    
    
def get_task(name):
    
    savefolder = r'C:\Users\lucam\Desktop\Learning\ReservoirImp\Sim\Bench'
    
    targets= np.load(os.path.join(savefolder,'all_targets.npy'),allow_pickle = True)
    names = np.load(os.path.join(savefolder,'target_names.npy'),allow_pickle = True)
    best_single = np.load(os.path.join(savefolder,'best_single.npy'),allow_pickle = True)
    best_fast_slow = np.load(os.path.join(savefolder,'best_fast_slow_narma_all.npy'),allow_pickle = True)
    best_slow_fast = np.load(os.path.join(savefolder,'best_slow_fast_narma_all.npy'),allow_pickle = True)
    best_3deep = np.load(os.path.join(savefolder,'best_deep3_narma_all.npy'),allow_pickle = True)

    names2 = names.tolist()
    ID = names2.index(name)

    target = targets[ID]
    single = best_single[ID]
    fast_slow = best_fast_slow[ID]
    slow_fast = best_slow_fast[ID]
    deep3 = best_3deep[ID]
    return target, single, fast_slow, slow_fast, deep3
        
        
        
        
class Adam:

    def __init__(self, Params):
        
        N_dim=np.shape(Params.shape)[0] # It finds out if the parameters given are in a vector (N_dim=1) or a matrix (N_dim=2)
        
        # INITIALISATION OF THE MOMENTUMS
        if N_dim==1:
               
            self.N1=Params.shape[0]
            
            self.mt=torch.zeros([self.N1])
            self.vt=torch.zeros([self.N1])
        
        if N_dim==2:
            
            self.N1=Params.shape[0]
            self.N2=Params.shape[1]
        
            self.mt=torch.zeros([self.N1,self.N2])
            self.vt=torch.zeros([self.N1,self.N2])
        
        # HYPERPARAMETERS OF ADAM
        self.beta1=0.9
        self.beta2=0.999
        
        self.epsilon=10**(-8)
        
        # COUNTER OF THE TRAINING PROCESS
        self.counter=0
        
        
    def Compute(self,Grads):
        
        # Compute the Adam updates by following the scheme above (beginning of the notebook)
        
        self.counter=self.counter+1
        
        self.mt=self.beta1*self.mt+(1-self.beta1)*Grads
        
        self.vt=self.beta2*self.vt+(1-self.beta2)*Grads**2
        
        mt_n=self.mt/(1-self.beta1**self.counter)
        vt_n=self.vt/(1-self.beta2**self.counter)
        
        New_grads=mt_n/(torch.sqrt(vt_n)+self.epsilon)
        
        return New_grads
        
        
        
           
                
        
        
        

        
        
        
        
    
    
    
    
    
    
                   
    