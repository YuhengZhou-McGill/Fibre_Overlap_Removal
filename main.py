import numpy as np 
from fibreGeneration_Full_Torch_PBC import fibreGeneration3DWithBoundaryTreatment
import time
import os
import random
import shutil



# %% main
for attempts in range(0,2):
    # %%% assign job id
    jobIDInt=random.randint(0, 100000000)
    jobID=str(jobIDInt)+'_'
    
    # %%% create work folder
    os.mkdir(str(jobIDInt))
    cwd = os.getcwd()
    shutil.copy("fibreGeneration_Full_Torch_PBC.py", str(jobIDInt))
    shutil.copy("minimumDistPara_GPU_Full_Torch.py", str(jobIDInt))
    shutil.copy("fibreInitialization.py", str(jobIDInt))
    shutil.copy("lengthDistribution.csv", str(jobIDInt))
    os.chdir(str(jobIDInt))

    # %%% initialize fibre position matrix
    if attempts ==0:
        # %%% load previous fibre sturcutre:
        fibrePositionMatrixPrevious=np.empty(0)
        FLPrevious = np.empty(0)
        FOThetaPrevious = np.empty(0)
        FOPhiPrevious = np.empty(0)
        
    # %%% define fibre statistics
    # %%%fibre
    if attempts==0:
        ASPLow=5
        ASPHigh=16
        orientationRangeTheta=1
        orientationRangePhi=1
        FRLow=0.0283
        FRHigh=0.0283
        numberOfFibres=20     
        # %%%% 5wt% 
        #numberOfFibres=24      
        # %%%% 10wt% 
        #numberOfFibres=45
        
    # %%%voids (change FR)
    if attempts==1:
        ASPLow=1e-9 
        ASPHigh=2e-9
        orientationRangeTheta=0
        orientationRangePhi=0
        FRLow=0.0333
        FRHigh=0.0333
        numberOfFibres=5
        #numberOfFibres=11
        
    # %%%% define maximum steps
    maxSteps=10000
    L,H,FR,FOThetaFull,FOPhiFull,FLFull,numberOfFibresFull,volumeFraction,aspectRatio,fibrePositionFull,fibrePositionMatrixFull,fibrePosition,FOTheta,FOPhi,FL,numberOfFibresInBound,fibrePositionMatrix,fibrePositionPaddedFull,stepsCounter=fibreGeneration3DWithBoundaryTreatment(numberOfFibres,ASPLow,ASPHigh,orientationRangeTheta,orientationRangePhi,FRLow,FRHigh,fibrePositionMatrixPrevious,FLPrevious,FOThetaPrevious,FOPhiPrevious,attempts)
    fibrePosition=fibrePosition.cpu().numpy()
    fibrePositionMatrixPrevious=fibrePosition
    FLPrevious=FL
    FOThetaPrevious=FOTheta
    FOPhiPrevious=FOPhi
    np.save(jobID+'fibrePositionPrevious.npy', fibrePosition)
    np.save(jobID+'FLPrevious.npy', FL)
    np.save(jobID+'FOThetaPrevious.npy', FOTheta)
    np.save(jobID+'FOPhiPrevious.npy', FOPhi)
    
    np.savetxt(jobID+'fibrePositionFull.csv',fibrePositionFull, delimiter=",")
    np.savetxt(jobID+'numberOfFibres.csv',np.array([numberOfFibresInBound]))
    np.savetxt(jobID+'fibrevolumeFraction.csv',np.array([volumeFraction]))
    np.savetxt(jobID+'fibreRadius.csv',np.array([FR]))
    
    # %%% save fibre distribtuion
    if attempts==1:
        np.savetxt(jobID+'fibrePositionFull.csv',fibrePositionFull, delimiter=",")
        np.savetxt(jobID+'numberOfFibres.csv',np.array([numberOfFibresInBound]))
        np.savetxt(jobID+'fibrevolumeFraction.csv',np.array([volumeFraction]))
        np.savetxt(jobID+'fibreRadius.csv',np.array([FR]))
    
    os.chdir(cwd)
