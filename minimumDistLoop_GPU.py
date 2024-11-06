import numpy as np
import torch
import time
import gc
import copy
import warnings
warnings.filterwarnings("ignore")

def minimumDistLoop(p1,p2,p3,p4,device):
    numberOfFibres=p1.size()[0]
    start_time = time.time()
    
    dP=torch.zeros(p1.shape, device=device).double()
    midPointCoord=torch.zeros(p1.shape, device=device).double()
    distance=torch.zeros(numberOfFibres, device=device).double()
    
    for counter in range(0,numberOfFibres):
    
        u = p1[counter,:] - p2[counter,:]
        v = p3[counter,:] - p4[counter,:]
        w = p2[counter,:] - p4[counter,:]
        
        if torch.sum(torch.isnan(p1)>0):
            print('1start')
            
        if torch.sum(torch.isnan(p2)>0):
            print('2start')
    
        if torch.sum(torch.isnan(p3)>0):
            print('3start')
        if torch.sum(torch.isnan(p4)>0):
            print('4start')
        
        a = torch.dot(u,u)
        b = torch.dot(u,v)
        c = torch.dot(v,v)
        d = torch.dot(u,w)
        e = torch.dot(v,w)
        
       
        
        D=copy.deepcopy(a*c-b*b)
        sD=copy.deepcopy(D)
        tD=copy.deepcopy(D)
    
        arraySize=D.shape
        tN=torch.zeros(arraySize, device=device).double()
        sN=torch.zeros(arraySize, device=device).double()
        sc=torch.zeros(arraySize, device=device).double()
        tc=torch.zeros(arraySize, device=device).double()
        
        
        smallNum=1e-9
        
        
        
        index=torch.where(D<smallNum)
        if any(map(len, index)):
            sN=0
            sD=1
            tN=e
            tD=c
        
        index=torch.where(D>=smallNum)
        if any(map(len, index)):
            sN=b*e-c*d
            tN=a*e-b*d
            
        subIndex=torch.where((D>=smallNum)&(sN<0))
        if any(map(len, subIndex)):
            sN = 0
            tN = e
            tD = c
            
        subIndex=torch.where((D>=smallNum)&(sN>0)&(sN>sD))
        if any(map(len, subIndex)):
            sN = sD
            tN = e + b
            tD = c
        
        
        
        index=torch.where(tN<0)
        if any(map(len, index)):
            tN=0
            
            subsN=sN
            subsD=sD
            suba = a
            subb = b
            subd = d
            
            subIndex=torch.where(-subd<0)
            if any(map(len, subIndex)):
                subsN=0
                
            subIndex=torch.where((-subd>=0)&(-subd>suba))
            if any(map(len, subIndex)):
                subsN=subsD
                
            subIndex=torch.where((-subd<=suba)&(-subd>=0))
            if any(map(len, subIndex)):
                subsN=-subd
                subsD=suba
                
            sN=subsN
            sD=subsD
        
        
        
        index=torch.where((tN>tD)&(tN>0))
        if any(map(len, index)):
            tN=tD
            
            subsN=sN
            subsD=sD
            suba = a
            subb = b
            subd = d
            
            subIndex=torch.where((-subd+subb)<0)
            if any(map(len, subIndex)):
                subsN=0
                
            subIndex=torch.where(((-subd+subb)>suba)&((-subd+subb)>=0))
            if any(map(len, subIndex)):
                subsN=subsD
                
            subIndex=torch.where(((-subd+subb)<=suba)&((-subd+subb)>=0))
            if any(map(len, subIndex)):
                subsN=-subd+subb
                subsD=suba
            
            sN=subsN
            sD=subsD        
        
        if ~torch.is_tensor(sN):
            sN=torch.tensor(sN)
            
        index=torch.where(torch.abs(sN)<smallNum)
        if any(map(len, index)):
            sc=0
        index=torch.where(torch.abs(sN)>=smallNum)
        if any(map(len, index)):
            sc=sN/sD
        
        if ~torch.is_tensor(tN):
            tN=torch.tensor(tN)
        index=torch.where(torch.abs(tN)<smallNum)
        if any(map(len, index)):
            tc=0
        index=torch.where(torch.abs(tN)>=smallNum)
        if any(map(len, index)):
            tc=tN/tD
        

        
        dPStep=w+(sc*u)-(tc*v)
        #dPStep=dPStep.reshape((len(dPStep),1))
        
        #print(dPStep.size())
        #print(w.size())
        
        dP[counter,:]=dPStep
        #dP[counter,:]=dPStep.reshape((len(dPStep),1))
        
        
        midPointCoord[counter,:]=p2[counter,:]+(sc*u)+0.5*dP[counter,:]
        
        distance[counter]=torch.linalg.norm(dP[counter,:])
        
        if ~torch.is_tensor(sc):
            sc=torch.tensor(sc)
            
        if ~torch.is_tensor(tc):
            tc=torch.tensor(tc)
        
        if torch.sum(torch.isnan(w)>0):
            print('w')
        if torch.sum(torch.isnan(sc)>0):
            print('sc')
            
        if torch.sum(torch.isnan(u)>0):
            print('u')
        if torch.sum(torch.isnan(tc)>0):
            print('tc')
        if torch.sum(torch.isnan(v)>0):
            print('v')
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    numberOfpairs=u.size()[0]
    #print(elapsed_time)
    #print(f"{elapsed_time} seconds to complete {device} distance calc.")
    #print(f"{numberOfpairs} pairs.")
    
    del u,v,w,p1,p2,p3,p4,device
    del a,b,c,d,e,D,sD,tD,dPStep
    del arraySize,tN,sN,sc,tc
    del index, subsN, subsD, suba, subb, subd, subIndex
    torch.cuda.empty_cache()
    gc.collect()
    
    return distance, dP ,midPointCoord