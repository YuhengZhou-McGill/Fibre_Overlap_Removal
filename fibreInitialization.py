import numpy as np 
import math


def sample_from_histogram(cdf, bin_edges, size=1000):
    # Generate uniform random numbers
    uniform_samples = np.random.rand(size)
    # Convert the uniform samples to the histogram's distribution
    bin_indices = np.searchsorted(cdf, uniform_samples)
    # Sample from the bin edges
    samples = bin_edges[bin_indices]
    return samples


def fibreInitializationWithDistribution(numberOfFibres,length,width,height,FRLow,FRHigh,ASPLow,ASPHigh,orientationRangeTheta,orientationRangePhi):
    #%% create fibres based on the fibre creation parameters
    #%%% define fibre radius
    FRMean=(FRLow+FRHigh)/2
    FRRange=(FRLow-FRHigh)/2
    randFactorsFR=(np.random.rand(numberOfFibres)-0.5)*2
    
    FR=FRMean+randFactorsFR*FRRange
    #%%% define fibre length based on distribution
    
    #get fibre length distribution
    fibreLengthDistribution=np.genfromtxt('lengthDistribution.csv', delimiter=',')
    #fibreLengthDistribution = np.random.uniform(0.1, 0.9, 1000)
    fibreLengthDistribution=fibreLengthDistribution*FRMean*2
    
    #histgram for fibre length distribution
    lengthBins = np.linspace(0.1, 1.3, 30)  # 30 bins covering the range from 0 to 90
    hist, bin_edges = np.histogram(fibreLengthDistribution, bins=lengthBins)
    
    #get the CDF of length distribution
    cdfLength = np.cumsum(hist)
    #normalize CDF
    cdfLength=cdfLength/cdfLength[-1]
    
    #sample fibre length based on the CDF
    lengthSample = sample_from_histogram(cdfLength, bin_edges, size=numberOfFibres)
    
    #save fibre distribution infomation
    np.save('fibreLengthDistribution.npy', fibreLengthDistribution)
    np.save('lengthSample.npy', lengthSample)
    np.save('lengthBins.npy', lengthBins)
    
    FL=np.zeros((3,numberOfFibres))
    FL[0,:]=lengthSample
    
    
    #%%% define fibre orientation
    
    #%%%% theta z-axis rotation
    #get theta distribution
    FOThetaDistribution = np.random.uniform(0, 90, 1000)
    
    #histgram for fibre length distribution
    FOThetaBins = np.linspace(0, 90, 30)  # 30 bins covering the range from 0 to 90
    hist, bin_edges = np.histogram(FOThetaDistribution, bins=FOThetaBins)
    
    #get the CDF of length distribution
    cdfFOTheta = np.cumsum(hist)
    #normalize CDF
    cdfFOTheta=cdfFOTheta/cdfFOTheta[-1]
    
    #sample fibre length based on the CDF
    FOThetaSample = sample_from_histogram(cdfFOTheta, bin_edges, size=numberOfFibres)
    
    #save fibre distribution infomation
    np.save('FOThetaDistribution.npy', FOThetaDistribution)
    np.save('FOThetaSample.npy', FOThetaSample)
    np.save('FOThetaBins.npy', FOThetaBins)
    
    FOTheta=FOThetaSample
    
    #%%%% phi y-axis rotation
    #get phi distribution
    FOPhiDistribution = np.random.uniform(0, 90, 1000)
    
    #histgram for fibre length distribution
    FOPhiBins = np.linspace(0, 90, 30)  # 30 bins covering the range from 0 to 90
    hist, bin_edges = np.histogram(FOPhiDistribution, bins=FOPhiBins)
    
    #get the CDF of length distribution
    cdfFOPhi = np.cumsum(hist)
    
    #normalize CDF
    cdfFOPhi=cdfFOPhi/cdfFOPhi[-1]
    
    #sample fibre length based on the CDF
    FOPhiSample = sample_from_histogram(cdfFOPhi, bin_edges, size=numberOfFibres)
    
    #save fibre distribution infomation
    np.save('FOPhiDistribution.npy', FOPhiDistribution)
    np.save('FOPhiSample.npy', FOPhiSample)
    np.save('FOPhiBins.npy', FOPhiBins)
    
    FOPhi=FOPhiSample
    
    #%%%rotate fibres based on the defined angles
    #%%%% z-axis rotation
    FOTheta=FOTheta*(math.pi/180.0)
    
    #%%%% y-axis rotation
    FOPhi=FOPhi*(math.pi/180.0)
    
    
    #%%%% define rotation tensor
    Rz11=np.cos(FOTheta).reshape((numberOfFibres,1))
    Rz12=-np.sin(FOTheta).reshape((numberOfFibres,1))
    Rz13=np.zeros((numberOfFibres,1))
    Rz21=np.sin(FOTheta).reshape((numberOfFibres,1))
    Rz22=np.cos(FOTheta).reshape((numberOfFibres,1))
    Rz23=np.cos(FOTheta).reshape((numberOfFibres,1))
    Rz31=np.zeros((numberOfFibres,1))
    Rz32=np.zeros((numberOfFibres,1))
    Rz33=np.ones((numberOfFibres,1))
    
    Ry11=np.cos(FOPhi).reshape((numberOfFibres,1))
    Ry12=np.zeros((numberOfFibres,1))
    Ry13=np.sin(FOPhi).reshape((numberOfFibres,1))
    Ry21=np.zeros((numberOfFibres,1))
    Ry22=np.ones((numberOfFibres,1))
    Ry23=np.zeros((numberOfFibres,1))
    Ry31=-np.sin(FOPhi).reshape((numberOfFibres,1))
    Ry32=np.zeros((numberOfFibres,1))
    Ry33=np.cos(FOPhi).reshape((numberOfFibres,1))
    
    rotationTensorTheta=np.zeros((3*numberOfFibres,3))
    rotationTensorTheta[::3,:]=np.concatenate((Rz11,Rz12,Rz13),axis=1)
    rotationTensorTheta[1::3,:]=np.concatenate((Rz21,Rz22,Rz23),axis=1)
    rotationTensorTheta[2::3,:]=np.concatenate((Rz31,Rz32,Rz33),axis=1)
    
    rotationTensorPhi=np.zeros((3*numberOfFibres,3))
    rotationTensorPhi[::3,:]=np.concatenate((Ry11,Ry12,Ry13),axis=1)
    rotationTensorPhi[1::3,:]=np.concatenate((Ry21,Ry22,Ry23),axis=1)
    rotationTensorPhi[2::3,:]=np.concatenate((Ry31,Ry32,Ry33),axis=1)
    
    #%%%% apply rotation tensor
    FLTheta=np.zeros((3,numberOfFibres))
    fibreRotationTheta=np.dot(rotationTensorTheta, FL)
    fibreRotationThetaX=fibreRotationTheta[::3].diagonal()
    fibreRotationThetaY=fibreRotationTheta[1::3].diagonal()
    fibreRotationThetaZ=fibreRotationTheta[2::3].diagonal()
    
    FLTheta[0,:]=fibreRotationThetaX
    FLTheta[1,:]=fibreRotationThetaY
    FLTheta[2,:]=fibreRotationThetaZ
    
    FLPhi=np.zeros((3,numberOfFibres))
    fibreRotationPhi=np.dot(rotationTensorPhi, FLTheta)
    fibreRotationPhiX=fibreRotationPhi[::3].diagonal()
    fibreRotationPhiY=fibreRotationPhi[1::3].diagonal()
    fibreRotationPhiZ=fibreRotationPhi[2::3].diagonal()
    
    FLPhi[0,:]=fibreRotationPhiX
    FLPhi[1,:]=fibreRotationPhiY
    FLPhi[2,:]=fibreRotationPhiZ
    

    #%% define fibre distribution
    startX=length*np.random.rand(numberOfFibres)*(1-(FR*2/length))+FR
    startY=width*np.random.rand(numberOfFibres)*(1-(FR*2/width))+FR
    startZ=height*np.random.rand(numberOfFibres)*(1-(FR*2/height))+FR
    
    endX=startX+fibreRotationPhiX
    endY=startY+fibreRotationPhiY
    endZ=startZ+fibreRotationPhiZ
    
    #%% create fibre storage
    
    fibrePosition=np.zeros((19,numberOfFibres))
    fibrePosition[0,:]=startX
    fibrePosition[1,:]=startY
    fibrePosition[2,:]=startZ
    fibrePosition[3,:]=endX
    fibrePosition[4,:]=endY
    fibrePosition[5,:]=endZ
    fibrePosition[12,:]=FL[0,:]
    fibrePosition[13,:]=FOTheta
    fibrePosition[14,:]=FOPhi
    fibrePosition[15,:]=0
    fibrePosition[16,:]=FR
    
    return fibrePosition






def fibreInitialization(numberOfFibres,length,width,height,FRLow,FRHigh,ASPLow,ASPHigh,orientationRangeTheta,orientationRangePhi):
    #%% create fibres based on the fibre creation parameters
    #%%% define fibre radius
    FRMean=(FRLow+FRHigh)/2
    FRRange=(FRLow-FRHigh)/2
    randFactorsFR=(np.random.rand(numberOfFibres)-0.5)*2
    
    FR=FRMean+randFactorsFR*FRRange
    
    #%%% define fibre length based on asp
    ASPMean=(ASPLow+ASPHigh)/2
    ASPRange=(ASPHigh-ASPLow)/2
    nominalFL=FR*2*ASPMean
    deltaFL=FR*2*ASPRange
    
    randFactorsFL=(np.random.rand(numberOfFibres)-0.5)*2
    FL=np.zeros((3,numberOfFibres))
    FL[0,:]=nominalFL+randFactorsFL*deltaFL
    
    #%%%rotate fibres based on the defined angles
    #%%%% z-axis rotation
    FOTheta=np.zeros((1,numberOfFibres))
    randFactorsTheta=(np.random.rand(numberOfFibres)-0.5)*2
    FOTheta=FOTheta+randFactorsTheta*orientationRangeTheta
    FOTheta=FOTheta*(math.pi/180.0)
    
    #%%%% y-axis rotation
    FOPhi=np.zeros((1,numberOfFibres))
    randFactorsPhi=(np.random.rand(numberOfFibres)-0.5)*2 
    FOPhi=FOPhi+randFactorsPhi*orientationRangePhi 
    FOPhi=FOPhi*(math.pi/180.0)
    
    
    #%%%% define rotation tensor
    Rz11=np.cos(FOTheta).reshape((numberOfFibres,1))
    Rz12=-np.sin(FOTheta).reshape((numberOfFibres,1))
    Rz13=np.zeros((numberOfFibres,1))
    Rz21=np.sin(FOTheta).reshape((numberOfFibres,1))
    Rz22=np.cos(FOTheta).reshape((numberOfFibres,1))
    Rz23=np.cos(FOTheta).reshape((numberOfFibres,1))
    Rz31=np.zeros((numberOfFibres,1))
    Rz32=np.zeros((numberOfFibres,1))
    Rz33=np.ones((numberOfFibres,1))
    
    Ry11=np.cos(FOPhi).reshape((numberOfFibres,1))
    Ry12=np.zeros((numberOfFibres,1))
    Ry13=np.sin(FOPhi).reshape((numberOfFibres,1))
    Ry21=np.zeros((numberOfFibres,1))
    Ry22=np.ones((numberOfFibres,1))
    Ry23=np.zeros((numberOfFibres,1))
    Ry31=-np.sin(FOPhi).reshape((numberOfFibres,1))
    Ry32=np.zeros((numberOfFibres,1))
    Ry33=np.cos(FOPhi).reshape((numberOfFibres,1))
    
    rotationTensorTheta=np.zeros((3*numberOfFibres,3))
    rotationTensorTheta[::3,:]=np.concatenate((Rz11,Rz12,Rz13),axis=1)
    rotationTensorTheta[1::3,:]=np.concatenate((Rz21,Rz22,Rz23),axis=1)
    rotationTensorTheta[2::3,:]=np.concatenate((Rz31,Rz32,Rz33),axis=1)
    
    rotationTensorPhi=np.zeros((3*numberOfFibres,3))
    rotationTensorPhi[::3,:]=np.concatenate((Ry11,Ry12,Ry13),axis=1)
    rotationTensorPhi[1::3,:]=np.concatenate((Ry21,Ry22,Ry23),axis=1)
    rotationTensorPhi[2::3,:]=np.concatenate((Ry31,Ry32,Ry33),axis=1)
    
    #%%%% apply rotation tensor
    FLTheta=np.zeros((3,numberOfFibres))
    fibreRotationTheta=np.dot(rotationTensorTheta, FL)
    fibreRotationThetaX=fibreRotationTheta[::3].diagonal()
    fibreRotationThetaY=fibreRotationTheta[1::3].diagonal()
    fibreRotationThetaZ=fibreRotationTheta[2::3].diagonal()
    
    FLTheta[0,:]=fibreRotationThetaX
    FLTheta[1,:]=fibreRotationThetaY
    FLTheta[2,:]=fibreRotationThetaZ
    
    FLPhi=np.zeros((3,numberOfFibres))
    fibreRotationPhi=np.dot(rotationTensorPhi, FLTheta)
    fibreRotationPhiX=fibreRotationPhi[::3].diagonal()
    fibreRotationPhiY=fibreRotationPhi[1::3].diagonal()
    fibreRotationPhiZ=fibreRotationPhi[2::3].diagonal()
    
    FLPhi[0,:]=fibreRotationPhiX
    FLPhi[1,:]=fibreRotationPhiY
    FLPhi[2,:]=fibreRotationPhiZ
    

    #%% define fibre distribution
    startX=length*np.random.rand(numberOfFibres)
    startY=width*np.random.rand(numberOfFibres)
    startZ=height*np.random.rand(numberOfFibres)
    
    endX=startX+fibreRotationPhiX
    endY=startY+fibreRotationPhiY
    endZ=startZ+fibreRotationPhiZ
    
    #%% create fibre storage
    
    fibrePosition=np.zeros((19,numberOfFibres))
    fibrePosition[0,:]=startX
    fibrePosition[1,:]=startY
    fibrePosition[2,:]=startZ
    fibrePosition[3,:]=endX
    fibrePosition[4,:]=endY
    fibrePosition[5,:]=endZ
    fibrePosition[12,:]=FL[0,:]
    fibrePosition[13,:]=FOTheta
    fibrePosition[14,:]=FOPhi
    fibrePosition[15,:]=0
    fibrePosition[16,:]=FR
    
    return fibrePosition