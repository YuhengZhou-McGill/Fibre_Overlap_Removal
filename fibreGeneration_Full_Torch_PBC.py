from minimumDistPara_GPU_Full_Torch import minimumDist
from periodicBoundaryTreatment import periodicBoundaryPadding
from fibreInitialization import fibreInitialization,fibreInitializationWithDistribution
import numpy as np 
import math
import time
import random
from scipy.spatial.transform import Rotation as R
import torch
import gc
import copy
import RBFs

def delete_rows(inputArray, rowToDelete, device):
    mask = torch.ones(inputArray.size(0), dtype=torch.bool, device=device)
    mask[rowToDelete] = False
    result = inputArray[mask]
    torch.cuda.empty_cache()
    
    del mask
    del rowToDelete
    del inputArray
    del device
    
    return result
    
def fibreGeneration3DWithBoundaryTreatment(numberOfFibres,ASPLow,ASPHigh,orientationRangeTheta,orientationRangePhi,FRLow,FRHigh,fibrePositionMatrixPrevious,FLPrevious,FOThetaPrevious,FOPhiPrevious,attempts):
    
    #%% define attempt loop counter
    attemptCounter=0
    stepsCounter=0
    #%% define RVE size
    L=1
    W=1
    H=1
    noIntersection=False
    
    #%% create fibres based on the fibre creation parameters/define fibre distribution/create fibre storage
    if attempts==0:
        fibrePosition=fibreInitializationWithDistribution(numberOfFibres,L,W,H,FRLow,FRHigh,ASPLow,ASPHigh,orientationRangeTheta,orientationRangePhi)
        
    if attempts==1:
        fibrePosition=fibreInitialization(numberOfFibres,L,W,H,FRLow,FRHigh,ASPLow,ASPHigh,orientationRangeTheta,orientationRangePhi)
    
    FL=fibrePosition[12,:]
    FOTheta=fibrePosition[13,:]
    FOPhi=fibrePosition[14,:]
    FR=fibrePosition[16,:]
    
    #%% define intersection removal parameters
    #%%% fibre distance/boundary threshold  
    margin=1
    
    #%%% define intersection removal parameters
    searchRadius=5
    stepLength=0.1
    
    #%%% define interection check parameters
    minIterseection=1e9 
    
    #%%% initalize iteration counter
    generationIterationCounter=0
    relocationIterationCounter=0
    
    #%% define boundary allowance
    if attempts==0:
        boundaryOffSetMargin=0.005
        xBoundaryLow=0+boundaryOffSetMargin
        xBoundaryHigh=L-boundaryOffSetMargin
        yBoundaryLow=0+boundaryOffSetMargin
        yBoundaryHigh=W-boundaryOffSetMargin
        zBoundaryLow=0+boundaryOffSetMargin
        zBoundaryHigh=H-boundaryOffSetMargin
        
    if attempts==1:
        boundaryOffSetMargin=0.005
        xBoundaryLow=0+boundaryOffSetMargin
        xBoundaryHigh=L-boundaryOffSetMargin
        yBoundaryLow=0+boundaryOffSetMargin
        yBoundaryHigh=W-boundaryOffSetMargin
        zBoundaryLow=0+boundaryOffSetMargin
        zBoundaryHigh=H-boundaryOffSetMargin

    #%% mark old fibres and combine previous fibres and new fibres
    if fibrePositionMatrixPrevious.any():
        fibrePositionMatrixPrevious[6,:]=0
        fibrePositionMatrixPrevious[7,:]=0
        fibrePositionMatrixPrevious[8,:]=0
        fibrePositionMatrixPrevious[9,:]=0
        fibrePositionMatrixPrevious[10,:]=0
        fibrePositionMatrixPrevious[11,:]=0
        fibrePositionMatrixPrevious[15,:]=1
        
        #combine previous fibres and new fibres
        fibrePosition=np.concatenate((fibrePositionMatrixPrevious,fibrePosition),axis=1)
        FL=np.concatenate((FLPrevious,FL),axis=0)
        FOTheta=np.concatenate((FOThetaPrevious,FOTheta),axis=0)
        FOPhi=np.concatenate((FOPhiPrevious,FOPhi),axis=0)
        numberOfFibresPrevious=fibrePositionMatrixPrevious.shape[1]
        numberOfFibres=numberOfFibres+numberOfFibresPrevious
        
        FR=fibrePosition[16,:]
        
    #%% define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device='cpu'
    torch.set_grad_enabled(False)
    fibrePosition=torch.from_numpy(fibrePosition).to(device)
    
    #%% initiate fibre distribution plot output
    fibrePositionPlot=np.zeros((500,fibrePosition.shape[0],5000))+10000
    intersectionCheckPlot=np.zeros((500,5000))
    
    #%% Keep the original fibre position matrix
    fibrePositionOrg=fibrePosition.clone().detach()
    
    
    #%% intersection optimization
    while noIntersection==False:
        start_time = time.time()
        
        #%% boundary treatment
        #%%% apply boundary offset
        fibrePositionFull=periodicBoundaryPadding(L,W,H,FR,fibrePosition.numpy())
        fibrePositionFull=torch.from_numpy(fibrePositionFull).to(device)
        
        #%%% total number of fibres
        numberOfFibresFull=fibrePositionFull.shape[1]
        
        #%% define non-intersecting fibre matrix
        nonintersectingFibres=torch.zeros(numberOfFibresFull)
        
        
        #%% fibre distance calculation
        #%%% get vectorization index
        upperIndex=torch.triu_indices(numberOfFibresFull,numberOfFibresFull, device=device)
        diagIndex=torch.where(upperIndex[0][:]==upperIndex[1][:])
        upperIndexNoDiagRow=delete_rows(upperIndex[0],diagIndex,device)
        upperIndexNoDiagColumn=delete_rows(upperIndex[1],diagIndex,device)
        upperIndexNoDiag=(upperIndexNoDiagRow,upperIndexNoDiagColumn)
        lowerIndexNoDiag=(upperIndexNoDiagColumn,upperIndexNoDiagRow)
        
        
        #%%% vectorization   
        L1=torch.repeat_interleave(fibrePositionFull[0:6,:].T, numberOfFibresFull, axis=0).reshape(numberOfFibresFull,numberOfFibresFull,6)
        L2=torch.vstack([fibrePositionFull[0:6,:].T]*numberOfFibresFull).reshape(numberOfFibresFull,numberOfFibresFull,6)
        
        L1start=L1[upperIndexNoDiag][:,0:3]
        L1end=L1[upperIndexNoDiag][:,3:6]
        L2start=L2[upperIndexNoDiag][:,0:3]
        L2end=L2[upperIndexNoDiag][:,3:6]
        
        #%%% initialize storage for fibre distance/ gradient
        fibrePositionMatrixFull=torch.zeros((numberOfFibresFull,numberOfFibresFull),dtype=torch.double, device=device)
        
        gradientMatrixFullx=torch.zeros((numberOfFibresFull,numberOfFibresFull),dtype=torch.double, device=device)
        gradientMatrixFully=torch.zeros((numberOfFibresFull,numberOfFibresFull),dtype=torch.double, device=device)
        gradientMatrixFullz=torch.zeros((numberOfFibresFull,numberOfFibresFull),dtype=torch.double, device=device)
        
        dPMidPointCoordx=torch.zeros((numberOfFibresFull,numberOfFibresFull),dtype=torch.double, device=device)
        dPMidPointCoordy=torch.zeros((numberOfFibresFull,numberOfFibresFull),dtype=torch.double, device=device)
        dPMidPointCoordz=torch.zeros((numberOfFibresFull,numberOfFibresFull),dtype=torch.double, device=device)
        
        
        #%%% distance/ gradient calculation - GPU
        distance,dP,dPMidPointCoord=minimumDist(L1start,L1end,L2start,L2end,device)
        stepsCounter=stepsCounter+1
        
        fibrePositionMatrixFull[upperIndexNoDiag]=distance
        fibrePositionMatrixFull[lowerIndexNoDiag]=fibrePositionMatrixFull[upperIndexNoDiag]
        np.save('fibrePositionMatrixFull.npy', fibrePositionMatrixFull)
        
        #%%% get shortest inter-fibre distance
        closestDistance, closestDistanceIndex=torch.sort(fibrePositionMatrixFull, 1)
        
        #%% inter-inclusion distance matrix
        FRFull=fibrePositionFull[16,:]
        np.save('FRFull.npy', FRFull)
        interInclusionDistanceMatrix=FRFull.repeat(numberOfFibresFull, 1)
        interInclusionDistanceMatrix=interInclusionDistanceMatrix+interInclusionDistanceMatrix.t()
        np.save('interInclusionDistanceMatrix.npy', interInclusionDistanceMatrix)
        #%%% apply margin
        interInclusionDistanceMatrix=interInclusionDistanceMatrix*margin
        
        
        #%%% check fibre intersection
        intersectionCheck=fibrePositionMatrixFull<interInclusionDistanceMatrix
        np.save('intersectionCheck.npy', intersectionCheck)
        
        
        #%%% gradient
        #%%%% gradient normalization
        dPx=torch.sign(dP[:,0])
        dPy=torch.sign(dP[:,1])
        dPz=torch.sign(dP[:,2])
        
        #%%% apply RBF
        #%%%% apply step length
        stepLength=1
        
        #%%%% calculate RBF values
        c = random.uniform(10, 25)
        distance_RBF=(RBFs.RBF_Gaussian(distance,c))
        
        #%%%% prevent zero denominator 
        dPScalingFactor=fibrePositionMatrixFull[upperIndexNoDiag]
        dPScalingFactor=dPScalingFactor+1e-8
        
        #%%%% normalization 
        gradientMatrixFullx[upperIndexNoDiag]=(dP[:,0]**2/dPScalingFactor**2)**(1/2)*dPx
        gradientMatrixFully[upperIndexNoDiag]=(dP[:,1]**2/dPScalingFactor**2)**(1/2)*dPy
        gradientMatrixFullz[upperIndexNoDiag]=(dP[:,2]**2/dPScalingFactor**2)**(1/2)*dPz
        
        #%%%% apply RBF
        gradientMatrixFullx[upperIndexNoDiag]=gradientMatrixFullx[upperIndexNoDiag]*distance_RBF
        gradientMatrixFully[upperIndexNoDiag]=gradientMatrixFully[upperIndexNoDiag]*distance_RBF
        gradientMatrixFullz[upperIndexNoDiag]=gradientMatrixFullz[upperIndexNoDiag]*distance_RBF
        
        gradientMatrixFullx[lowerIndexNoDiag]=-gradientMatrixFullx[upperIndexNoDiag]
        gradientMatrixFully[lowerIndexNoDiag]=-gradientMatrixFully[upperIndexNoDiag]
        gradientMatrixFullz[lowerIndexNoDiag]=-gradientMatrixFullz[upperIndexNoDiag]
        overall=gradientMatrixFullx**2+gradientMatrixFully**2+gradientMatrixFullz**2
        


        #%%% fibre intersections count
        
        noIntersection=torch.sum(intersectionCheck)==numberOfFibresFull
        
        intersectingFibres=torch.sum(intersectionCheck,0)>1
        
        nonintersectingFibres=nonintersectingFibres-(intersectingFibres.int()-1)

#%%% chceck the shortest fibre to move
        if noIntersection==False:
            fibreRemovalMatrix=torch.zeros(4,numberOfFibresFull)
            fibreRemovalMatrix[0,:]=fibrePositionMatrixFull[12,:].to(device)
            fibreRemovalMatrix[1,:]=fibrePositionMatrixFull[13,:].to(device)
            fibreRemovalMatrix[2,:]=fibrePositionMatrixFull[14,:].to(device)
            fibreRemovalMatrix[3,:]=intersectingFibres
            
            maskIntersectingFibres=fibreRemovalMatrix[3,:]==1
            sorted_values, sorted_indices=torch.sort(fibreRemovalMatrix[0,:][maskIntersectingFibres], dim=0)
            
            shortestIntersectingFibreIndex=sorted_indices[0]
            shortestIntersectingFibreIndexGlobal = torch.where(maskIntersectingFibres)[0][shortestIntersectingFibreIndex]
            
            sortedIntersectingFibreIndex=sorted_indices[1:]
            sortedIntersectingFibreIndexGlobal = torch.where(maskIntersectingFibres)[0][sortedIntersectingFibreIndex]
        
        #%%% output each generation steps
        fibrePositionPlot[generationIterationCounter,:,0:fibrePositionFull.shape[1]]=fibrePositionFull
        np.save('fibrePositionPlot.npy', fibrePositionPlot)
        
        intersectionCheckPlot[generationIterationCounter,0:intersectingFibres.shape[0]]=intersectingFibres
        np.save('intersectionCheckPlot.npy', intersectionCheckPlot)
        
        
        #%%% seprate new and previous fibres
        newFibresCheck=fibrePositionFull[15,:]==0
        
        
        #%%%apply RBF to movement
        #%%% stochastic gradient descent
        gradOff=torch.rand((numberOfFibresFull, numberOfFibresFull))
        gradOff= (gradOff >= 0.005).int()
        
        #%%% define fibre movement (fix fibres based on marks)
        #define movement for each fibre
        xMovement=torch.sum(gradientMatrixFullx*(FRFull)*gradOff, 1, keepdim=False)*intersectingFibres*newFibresCheck
        yMovement=torch.sum(gradientMatrixFully*(FRFull)*gradOff, 1, keepdim=False)*intersectingFibres*newFibresCheck
        zMovement=torch.sum(gradientMatrixFullz*(FRFull)*gradOff, 1, keepdim=False)*intersectingFibres*newFibresCheck
        
        #%%% apply movement to fibres (define movement function)
        if noIntersection==False:
            #%%%% copy previous fibre locations
            fibrePositionPrevious=copy.deepcopy(fibrePosition.detach())
            
            #%%%% apply movement function
            fibrePosition[0,:]=fibrePosition[0,:]+xMovement[0:numberOfFibres]
            fibrePosition[3,:]=fibrePosition[3,:]+xMovement[0:numberOfFibres]
            fibrePosition[1,:]=fibrePosition[1,:]+yMovement[0:numberOfFibres]
            fibrePosition[4,:]=fibrePosition[4,:]+yMovement[0:numberOfFibres]
            fibrePosition[2,:]=fibrePosition[2,:]+zMovement[0:numberOfFibres]
            fibrePosition[5,:]=fibrePosition[5,:]+zMovement[0:numberOfFibres]
            
            
            
            #%%% check fibres moved out from the domain
            #%%%% first point check
            #%%%%% x boundary
            xBoundaryLowCheckFirst=fibrePosition[0,:]>xBoundaryLow
            xBoundaryHighCheckFirst=fibrePosition[0,:]<xBoundaryHigh
            #%%%%% y boundary
            yBoundaryLowCheckFirst=fibrePosition[1,:]>yBoundaryLow
            yBoundaryHighCheckFirst=fibrePosition[1,:]<yBoundaryHigh
            #%%%%% z boundary
            zBoundaryLowCheckFirst=fibrePosition[2,:]>zBoundaryLow
            zBoundaryHighCheckFirst=fibrePosition[2,:]<zBoundaryHigh
            
            #%%%% second point check
            #%%%%% x boundary
            xBoundaryLowCheckSecond=fibrePosition[3,:]>xBoundaryLow
            xBoundaryHighCheckSecond=fibrePosition[3,:]<xBoundaryHigh
            
            #%%%%% y boundary
            yBoundaryLowCheckSecond=fibrePosition[4,:]>yBoundaryLow
            yBoundaryHighCheckSecond=fibrePosition[4,:]<yBoundaryHigh
            
            #%%%%% z boundary
            zBoundaryLowCheckSecond=fibrePosition[5,:]>zBoundaryLow
            zBoundaryHighCheckSecond=fibrePosition[5,:]<zBoundaryHigh
            
            #%%%% index of out-of-boundary fibres
            xboundaryCheckAfterMove=(xBoundaryLowCheckFirst*xBoundaryHighCheckFirst)+(xBoundaryLowCheckSecond*xBoundaryHighCheckSecond)
            yboundaryCheckAfterMove=(yBoundaryLowCheckFirst*yBoundaryHighCheckFirst)+(yBoundaryLowCheckSecond*yBoundaryHighCheckSecond)
            zboundaryCheckAfterMove=(zBoundaryLowCheckFirst*zBoundaryHighCheckFirst)+(zBoundaryLowCheckSecond*zBoundaryHighCheckSecond)
            
            xindexForPullBack=xboundaryCheckAfterMove<1
            yindexForPullBack=yboundaryCheckAfterMove<1
            zindexForPullBack=zboundaryCheckAfterMove<1
            
            
            xindexForRemain=-(xindexForPullBack.int()-1)
            yindexForRemain=-(yindexForPullBack.int()-1)
            zindexForRemain=-(zindexForPullBack.int()-1)
            
            indexForPullBackAll=(xindexForPullBack+yindexForPullBack+zindexForPullBack)>0
            numberOfPullBackFibres=torch.sum(indexForPullBackAll)
            
            #%%%% push back the out of boundary fibres
            fibrePosition[0,:]=fibrePositionPrevious[0,:]*xindexForPullBack+fibrePosition[0,:]*xindexForRemain
            fibrePosition[3,:]=fibrePositionPrevious[3,:]*xindexForPullBack+fibrePosition[3,:]*xindexForRemain
            
            fibrePosition[1,:]=fibrePositionPrevious[1,:]*yindexForPullBack+fibrePosition[1,:]*yindexForRemain
            fibrePosition[4,:]=fibrePositionPrevious[4,:]*yindexForPullBack+fibrePosition[4,:]*yindexForRemain
            
            fibrePosition[2,:]=fibrePositionPrevious[2,:]*zindexForPullBack+fibrePosition[2,:]*zindexForRemain
            fibrePosition[5,:]=fibrePositionPrevious[5,:]*zindexForPullBack+fibrePosition[5,:]*zindexForRemain
            
            
            #%% define fibre relocation
            if (stepsCounter+1)%500==0:
                
                #%%% move the shortest fibre 
                
                fibrePositionReplacement = fibreInitialization(1,L,W,H,FR,ASPLow,ASPHigh,orientationRangeTheta,orientationRangePhi)
                
                fibrePosition[:,shortestIntersectingFibreIndexGlobal:shortestIntersectingFibreIndexGlobal+1]=torch.from_numpy(fibrePositionReplacement).to(device)
                
                nonintersectingFibres=torch.zeros(numberOfFibresFull)
                
            #%%% zero duplicated fibre index
            fibrePosition[6:12]=0
            #%%% count the generation steps
            generationIterationCounter=generationIterationCounter+1
            
            #%%% timer
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time  
            
            #%%% clean up
            del fibrePositionFull,numberOfFibresFull,upperIndex,diagIndex,upperIndexNoDiagRow,upperIndexNoDiagColumn,upperIndexNoDiag,lowerIndexNoDiag
            del L1,L2,L1start,L1end,L2start,L2end
            del fibrePositionMatrixFull,gradientMatrixFullx,gradientMatrixFully,gradientMatrixFullz,dPMidPointCoordx,dPMidPointCoordy,dPMidPointCoordz
            del distance,dP,dPMidPointCoord,closestDistance,closestDistanceIndex,intersectionCheck
            del dPx,dPy,dPz,dPScalingFactor,overall
            
            if 'fibrePositionReplacement' in locals():
                del fibrePositionReplacement
            
            if 'boundaryCheckFirstPoint' in locals():
                del boundaryCheckFirstPoint
                
            if 'boundaryCheckSecondPoint' in locals():
                del boundaryCheckSecondPoint
            
            if 'indexForPullBackColumn' in locals():
                del indexForPullBackColumn
            
            torch.cuda.empty_cache()
            gc.collect()
    #%% output fibre data
    #%%% convert data type to np
    fibrePositionFull=fibrePositionFull.cpu().numpy()
    fibrePositionMatrixFull=fibrePositionMatrixFull.cpu().numpy()
    intersectionCheck=intersectionCheck.cpu().numpy()
    #%%% output fibre structure 
    volumeFraction=np.sum(FL*FR*FR*math.pi)/(L*W*H)
    aspectRatio=FL/(FR*2)
    FLFull=np.zeros((3,numberOfFibresFull))
    FLFull[0,:]=fibrePositionFull[12,:]
    FOThetaFull=fibrePositionFull[13,:]
    FOPhiFull=fibrePositionFull[14,:]
    fibrePositionMatrix=fibrePositionFull
    fibrePositionPaddedFull=fibrePositionFull
    
    return L,H,FR,FOThetaFull,FOPhiFull,FLFull,numberOfFibresFull,volumeFraction,aspectRatio,fibrePositionFull,fibrePositionMatrixFull,fibrePosition,FOTheta,FOPhi,FL,numberOfFibres,fibrePositionMatrix,fibrePositionPaddedFull,stepsCounter