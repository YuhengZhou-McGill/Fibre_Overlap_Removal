import numpy as np
import torch
import time
import gc
import copy

def minimumDist(p1,p2,p3,p4,device):
    
    start_time = time.time()
    
    u = p1 - p2
    v = p3 - p4
    w = p2 - p4
    
    if torch.sum(torch.isnan(p1)>0):
        print('1start')
        
    if torch.sum(torch.isnan(p2)>0):
        print('2start')

    if torch.sum(torch.isnan(p3)>0):
        print('3start')
        
    if torch.sum(torch.isnan(p4)>0):
        print('4start')
    
    a = torch.einsum('ij,ij->i',u,u)
    b = torch.einsum('ij,ij->i',u,v)
    c = torch.einsum('ij,ij->i',v,v)
    d = torch.einsum('ij,ij->i',u,w)
    e = torch.einsum('ij,ij->i',v,w)
    
    
    a = a.reshape((len(a),1))
    b = b.reshape((len(b),1))
    c = c.reshape((len(c),1))
    d = d.reshape((len(d),1))
    e = e.reshape((len(e),1))
   
    
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
        sN[index]=0
        sD[index]=1
        tN[index]=e[index]
        tD[index]=c[index]
    
    index=torch.where(D>=smallNum)
    if any(map(len, index)):
        sN[index]=b[index]*e[index]-c[index]*d[index]
        tN[index] =a[index]*e[index]-b[index]*d[index]
        
    subIndex=torch.where((D>=smallNum)&(sN<0))
    if any(map(len, subIndex)):
        sN[subIndex] = 0.0
        tN[subIndex] = e[subIndex]
        tD[subIndex] = c[subIndex]
        
    subIndex=torch.where((D>=smallNum)&(sN>0)&(sN>sD))
    if any(map(len, subIndex)):
        sN[subIndex] = sD[subIndex]
        tN[subIndex] = e[subIndex] + b[subIndex]
        tD[subIndex] = c[subIndex]
    
    
    
    index=torch.where(tN<0)
    if any(map(len, index)):
        tN[index]=0
        
        subsN=sN[index]
        subsD=sD[index]
        suba = a[index]
        subb = b[index]
        subd = d[index]
        
        subIndex=torch.where(-subd<0)
        if any(map(len, subIndex)):
            subsN[subIndex]=0
            
        subIndex=torch.where((-subd>=0)&(-subd>suba))
        if any(map(len, subIndex)):
            subsN[subIndex]=subsD[subIndex]
            
        subIndex=torch.where((-subd<=suba)&(-subd>=0))
        if any(map(len, subIndex)):
            subsN[subIndex]=-subd[subIndex]
            subsD[subIndex]=suba[subIndex]
            
        sN[index]=subsN
        sD[index]=subsD
    
    
    
    index=torch.where((tN>tD)&(tN>0))
    if any(map(len, index)):
        tN[index]=tD[index]
        
        subsN=sN[index]
        subsD=sD[index]
        suba = a[index]
        subb = b[index]
        subd = d[index]
        
        subIndex=torch.where((-subd+subb)<0)
        if any(map(len, subIndex)):
            subsN[subIndex]=0
            
        subIndex=torch.where(((-subd+subb)>suba)&((-subd+subb)>=0))
        if any(map(len, subIndex)):
            subsN[subIndex]=subsD[subIndex]
            
        subIndex=torch.where(((-subd+subb)<=suba)&((-subd+subb)>=0))
        if any(map(len, subIndex)):
            subsN[subIndex]=-subd[subIndex]+subb[subIndex]
            subsD[subIndex]=suba[subIndex]
        
        sN[index]=subsN
        sD[index]=subsD        
        
    
    
    index=torch.where(torch.abs(sN)<smallNum)
    if any(map(len, index)):
        sc[index]=0
    index=torch.where(torch.abs(sN)>=smallNum)
    if any(map(len, index)):
        sc[index]=sN[index]/sD[index]
    
    index=torch.where(torch.abs(tN)<smallNum)
    if any(map(len, index)):
        tc[index]=0
    index=torch.where(torch.abs(tN)>=smallNum)
    if any(map(len, index)):
        tc[index]=tN[index]/tD[index]
        
    dP=w+(sc*u)-(tc*v)
    
    midPointCoord=p2+(sc*u)+0.5*dP
    
    distance=torch.linalg.norm(dP,axis=1)
    
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
    del a,b,c,d,e,D,sD,tD
    del arraySize,tN,sN,sc,tc
    del index, subsN, subsD, suba, subb, subd, subIndex
    torch.cuda.empty_cache()
    gc.collect()
    
    return distance, dP ,midPointCoord