import numpy as np
import os 
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

device='cuda'

###################################################################
## METAL LEARNING CLASS FOR THE READ-OUT FROM THE DEVICES ACTIVITIES

class Meta_ReadOut(nn.Module):
    
    def __init__(self,Ns,t_in,eta):
        super().__init__()
        
        ## Number of layers in the network...it is always 1 for the case studeied, given that 
        ## are training a read-out from the devices responses
        N_layers=np.shape(Ns)[0]
        
        ## Lists of parameters
        self.Ws=[]
        self.bs=[]
        
        self.eta=eta
        
        self.t_in=t_in
        self.etas_bs=[]
        self.etas_Ws=[]
                
        self.Ws.append(nn.Parameter(torch.rand([Ns[0],Ns[1]],device=device)/(Ns[0]+Ns[1])))
        
        ## Multiplicative factor to each separate output weight matrix
        ## these are the variable that will change in the inner meta-learning loop
        self.W_out=[nn.Parameter(torch.ones([Ns[1]],device=device)*0.01)]
        
        self.bs=[nn.Parameter(torch.zeros([Ns[1]],device=device))]
        
        
        self.loss=nn.MSELoss()
        self.Ns=Ns
        
        
    def Initialise_Hyperparameters(self,eta_Meta,c,N_iter,t_in):
        
        
        ## Hyperparameterers
        self.eta_Meta=eta_Meta ## Learning rate for the Meta-update
        
        self.opt=optim.Adam([{ 'params': self.Ws, 'lr':eta_Meta }])
        self.c=c
        
        self.N_iter=N_iter 
        self.E=torch.zeros([N_iter,t_in]) ## Inner loop loss function
        self.L=torch.zeros([N_iter,t_in]) ## Outer loop (meta) loss function
        self.counter=0 
        self.w_t=torch.ones([t_in]) ## Weights to the different loss functions, computed at different steps of the 
                                    ## inner loop.
                                    ## In our simulations, it will be always zero but for the last step
        self.t_steps=t_in
        
    
    
    ## Computation of the output activities
    def Forward(self, S, Target, Ws, bs):
        
        xs=[]
        xs.append(S)
        
        Out= torch.add(torch.matmul(xs[0],Ws[0])*Ws[1],bs[0])
                
        E= self.loss(Out,Target)
                            
        return E, Out, xs[-1]
    
    
    ## Meta-learning step
    def Meta(self, S_T, T_T, S_q, T_q, t_steps):
        
        N_task=len(S_T)
        
        # Initialisation of the loss functions, where N_task is the number of tasks considered 
        # for training and t_steps are the number of steps accomplished in the inner loop
        E_n=torch.zeros([N_task,t_steps])
        L=torch.zeros([N_task,t_steps])
            
        W_out=self.W_out
        
        ## For each different task
        for n in range(N_task):
            
            ## First step
            E,_,_=self.Forward(S_T[n],T_T[n], self.Ws+W_out, self.bs) ## Inner loss function
        
            Grads_W=torch.autograd.grad(E, self.W_out, retain_graph=True)  ## Corresponding gradients
            
            W_out = list(map(lambda p: p[1] - self.eta * p[0], zip(Grads_W, W_out) )) ## Update of the output scaling factors
            
            L[n,0],_,_=self.Forward(S_q[n],T_q[n], self.Ws+W_out, self.bs) ## Compuation of the meta objective
            E_n[n,0]=E
            
            ## Other steps of the inner loop
            for t in range(1,t_steps):
                
                E,_,_=self.Forward(S_T[n],T_T[n], self.Ws+W_out, self.bs)   ## Same as before
        
                Grads_W=torch.autograd.grad(E, W_out, retain_graph=True)
                                
                W_out = list(map(lambda p: p[1] - self.eta * p[0], zip(Grads_W, W_out) ))

                L[n,t],_,_=self.Forward(S_q[n],T_q[n], self.Ws+W_out, self.bs)
                E_n[n,t]=E
        
        
        self.E[self.counter,:]=torch.mean(E_n.detach(),0)
        self.L[self.counter,:]=torch.mean(L.detach(),0)
        
        Train=self.Check_Errors()
        self.counter+=1
        
        No_Meta=False ## If this is True, it will perform simple perform gradient descent on the Error function 
                      ## in the first step
        
        
        if No_Meta:
            
            E=torch.sum(E_n[n,0])/N_task
            E.backward()
            
            self.opt.step()
            
            self.opt.zero_grad()
            
        else:
            
            E_meta=torch.sum(L*self.w_t.unsqueeze(0))/N_task
            E_meta.backward()
                        
            self.opt.step()
            self.opt.zero_grad()
        
        return E_n, L
    
    ## Class used to update the scaling factors of the error functions computed at the different steps of the inner loop
    ## In general, these factors can decay as explained in the paper "How to train your MAML". 
    ## In our setting, given that we are exploiting a first order MAML on a simple read-out, we will set only the
    ## final scaling factor to 1, and the resto to 0
    def Check_Errors(self):
        
        Train=True
        N_change=0
        if self.counter>=N_change:
            self.w_t[0:self.t_steps-1]=self.w_t[0:self.t_steps-1]*0 
        
        return Train
    
    ## Final few-shot learning adaptation
    def Fine_Tuning(self, S, Target, n_steps, x_data, y_data):
        
        E_n=torch.zeros([self.t_in+n_steps])
        E_all=torch.zeros([self.t_in+n_steps])
        Y_data=torch.zeros([self.t_in+n_steps,x_data.size()[0],self.Ns[-1]]).to('cpu')
        
        W_out=self.W_out
        Ws=self.Ws
        
        for n in range(self.t_in):
            
            
            e_all,y,z=self.Forward(x_data, y_data, Ws+W_out, self.bs)
            
            Y_data[n,:,:]=y.detach().to('cpu')
            
            E,_,_=self.Forward(S,Target, self.Ws+W_out, self.bs)

            Grads_W=torch.autograd.grad(E, W_out)
                        
            W_out = list(map(lambda p: p[1] - self.eta * p[0], zip(Grads_W, W_out)))
            
            E_n[n]=E.detach()
            E_all[n]=e_all.detach()
        
            
        for n in range(self.t_in,n_steps+self.t_in):
            
            e_all,y,z=self.Forward(x_data, y_data, Ws+W_out, self.bs)
            
            Y_data[n,:,:]=y.detach().to('cpu')
            
            E,_,_=self.Forward(S,Target, Ws+W_out, self.bs)

            Grads_W=torch.autograd.grad(E, Ws)
                        
            Ws = list(map(lambda p: p[1] - 0.0002 * p[0], zip(Grads_W, Ws)))
            
            E_n[n]=E.detach()
            E_all[n]=e_all.detach()
        
        return E_n, E_all, Y_data
    
    
    def Analysis(self, Ss, Targets, n_steps, x_datas, y_datas):
        
        N_task=len(Ss)
        
        YS=torch.zeros([N_task,self.t_in+n_steps,x_datas[0].size()[0],self.Ns[-1]]).to('cpu')
        ES=torch.zeros([N_task,self.t_in+n_steps]).to('cpu')
        ES_all=torch.zeros([N_task,self.t_in+n_steps]).to('cpu')
        
        for n in range(N_task):
            
            ES[n,:], ES_all[n,:], YS[n,:,:,:]= self.Fine_Tuning(Ss[n], Targets[n], n_steps, x_datas[n], y_datas[n])

            
        return ES, ES_all, YS    
    


####################################
### Data Manager to generate the data 

class SineWaves:
    
    def __init__(self,dts,X):
        
        self.N_t=dts.size()[0]
        self.dts=dts
        
        ## Hyperparameters defining the tasks distribution
        ## Range of amplitudes used
        self.As=[[-1.2,1.2],[-1.2,1.2],[-1.2,1.2],[-1.2,1.2],[-1.2,1.2]]
        
        ## Range of frequencies used
        self.phases=[[0,np.pi],[0,np.pi/2],[0,np.pi/2],[0,np.pi/3],[0,np.pi/4]]
        
        self.N_F=5
        self.X=X
    
    
    ## The method samples data fot the updates for a specific task, where batch_size1 (batch_size2) is the
    ## number of data points to update the parameters in the inner loop (for the meta-objective)
    ## if Rand=True, the training data are sampled randomly, if Rand=False, the training data will be equally spaced
    ## by the value in space
    def Sample(self,batch_size1,batch_size2,space,Rand):
        
        
        a=torch.zeros([self.N_F])
        theta=torch.zeros([self.N_F])
        
        
        ## Selecting the index of the data to be used
        if Rand==True:
            
            n_t=np.random.randint(0,self.X.size()[0],batch_size1)
            
        else:
            
            lin=np.arange(0,batch_size1)*space

            n_t=np.random.randint(0,self.X.size()[0]-lin[-1])+lin

        ns=np.delete(np.arange(0,self.X.size()[0]),n_t)
        
        ns=np.random.permutation(ns)
        
        ## Sampling the corresponding data, Xs_t, Xs_q
        xs_t=self.dts[n_t]
        Xs_t=self.X[n_t,:]
                
        n_q=ns[0:batch_size2]
        xs_q=self.dts[n_q]
        Xs_q=self.X[n_q,:]
        
        s_t=torch.zeros([batch_size1,self.N_F])
        s_q=torch.zeros([batch_size2,self.N_F])
        
        ## Sampling a task and the corresponding targets
        for n in range(self.N_F):
            
            
            a[n]=self.As[n][0]+torch.rand(1)*(self.As[n][1]-self.As[n][0])            
            theta[n]=self.phases[n][0]+torch.rand(1)*(self.phases[n][1]-self.phases[n][0])
        
            s_t[:,n]=a[n]*(torch.sin((n+1)*xs_t+theta[n])+1)/2
            s_q[:,n]=a[n]*(torch.sin((n+1)*xs_q+theta[n])+1)/2
            
        
        return s_t.to('cuda'), Xs_t.to('cuda'), n_t, s_q.to('cuda'), Xs_q.to('cuda'), n_q, a, theta
    
    ## The following method samples data for N_task 
    def Sample_Tr(self,N_task,batch_size1,batch_size2,space,Rand):
        
        T_T=[]
        S_T=[]
        T_q=[]
        S_q=[]
        N_T=[]
        N_Q=[]
        As=[]
        Ths=[]
        
        for n in range(N_task):
            
            s_t,xs_t,n_t,s_q,xs_q,n_q,a,theta=Sines.Sample(batch_size1,batch_size2,space,Rand)
            
            T_T.append(s_t)
            S_T.append(xs_t)
            T_q.append(s_q)
            S_q.append(xs_q)
            
            N_T.append(n_t)
            N_Q.append(n_q)
            
            As.append(a)
            Ths.append(theta)

            
        return T_T, S_T, N_T, T_q, S_q, N_Q, As, Ths
    
    ## Sample a task with the given values a for the amplitude and theta for the phase
    def Sample_1(self,a,theta,batch_size,space,Rand=True):
        
        if Rand==True:
            
            n_t=np.random.randint(0,self.X.size()[0],batch_size)
            
        else:
            
            lin=np.arange(0,batch_size)*space

            n_t=np.random.randint(0,self.X.size()[0]-lin[-1])+lin

        S_T=self.X[n_t,:]
        T_T=torch.zeros([batch_size,self.N_F])
        y_data=torch.zeros([self.X.size()[0],self.N_F])
        
        for n in range(self.N_F):
        
            T_T[:,n]=a[n]*(torch.sin((n+1)*self.dts[n_t]+theta[n])+1)/2
            y_data[:,n]=a[n]*(torch.sin((n+1)*self.dts+theta[n])+1)/2
            

        x_data=self.X[:,:]
        
        return S_T, T_T.to('cuda'), x_data, y_data.to('cuda'), n_t
    
    def sample_all(self,N_task,batch_size,space,Rand):
        
        Ss=[]
        Targets=[]
        x_datas=[]
        y_datas=[]
            
        N_T=[]
        As=[]
        Ths=[]
        Os=[]
        
        for n in range(N_task):
            
            a=torch.zeros([self.N_F])
            theta=torch.zeros([self.N_F])
            
            for l in range(self.N_F):
            
                a[l]=self.As[l][0]+torch.rand(1)*(self.As[l][1]-self.As[l][0])            
                theta[l]=self.phases[l][0]+torch.rand(1)*(self.phases[l][1]-self.phases[l][0])
        
            S_T, T_T, x_data, y_data, n_t=self.Sample_1(a,theta,batch_size,space,Rand)
            
            Ss.append(S_T)
            Targets.append(T_T)
            x_datas.append(x_data)
            y_datas.append(y_data)
            N_T.append(n_t)
            As.append(a)
            Ths.append(theta)
        
        return Ss, Targets, N_T, x_datas, y_datas, As, Ths
