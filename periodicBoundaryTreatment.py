import numpy as np 

def periodicBoundaryPadding(L,W,H,FR,fibrePosition):
        #%% boundary treatment
        #%%% retrieve boundary fibres 
        duplicatedFront=np.copy(fibrePosition)
        duplicatedBack=np.copy(fibrePosition)
        duplicatedRight=np.copy(fibrePosition)
        duplicatedLeft=np.copy(fibrePosition)
        duplicatedTop=np.copy(fibrePosition)
        duplicatedBottom=np.copy(fibrePosition)
        
        #%%% check fibre penetration (FR is included)
# =============================================================================
#         boundaryCheckFront=np.where(np.sum(fibrePosition[[0,3],:]>L,axis=0)>0)
#         boundaryCheckBack=np.where(np.sum(fibrePosition[[0,3],:]<0,axis=0)>0)
#         boundaryCheckRight=np.where(np.sum(fibrePosition[[1,4],:]<0,axis=0)>0)
#         boundaryCheckLeft=np.where(np.sum(fibrePosition[[1,4],:]>W,axis=0)>0)
#         boundaryCheckTop=np.where(np.sum(fibrePosition[[2,5],:]>H,axis=0)>0)
#         boundaryCheckBottom=np.where(np.sum(fibrePosition[[2,5],:]<0,axis=0)>0)
# =============================================================================
        
        #%%% apply boundary offset
        #%%%% Back     
        duplicatedBack1=np.copy(duplicatedBack)
        duplicatedBack4=np.copy(duplicatedBack)
        duplicatedBack7=np.copy(duplicatedBack)
        duplicatedBack10=np.copy(duplicatedBack)
        duplicatedBack13=np.copy(duplicatedBack)
        duplicatedBack16=np.copy(duplicatedBack)
        duplicatedBack19=np.copy(duplicatedBack)
        duplicatedBack22=np.copy(duplicatedBack)
        duplicatedBack25=np.copy(duplicatedBack)
        
        duplicatedBack1[[0,3],:]=duplicatedBack1[[0,3],:]+L
        duplicatedBack1[[1,4],:]=duplicatedBack1[[1,4],:]+W
        duplicatedBack1[[2,5],:]=duplicatedBack1[[2,5],:]+H
        duplicatedBack4[[0,3],:]=duplicatedBack4[[0,3],:]+L
        duplicatedBack4[[1,4],:]=duplicatedBack4[[1,4],:]+0
        duplicatedBack4[[2,5],:]=duplicatedBack4[[2,5],:]+H
        duplicatedBack7[[0,3],:]=duplicatedBack7[[0,3],:]+L
        duplicatedBack7[[1,4],:]=duplicatedBack7[[1,4],:]-W
        duplicatedBack7[[2,5],:]=duplicatedBack7[[2,5],:]+H
        
        duplicatedBack10[[0,3],:]=duplicatedBack10[[0,3],:]+L
        duplicatedBack10[[1,4],:]=duplicatedBack10[[1,4],:]+W
        duplicatedBack10[[2,5],:]=duplicatedBack10[[2,5],:]+0
        duplicatedBack13[[0,3],:]=duplicatedBack13[[0,3],:]+L
        duplicatedBack13[[1,4],:]=duplicatedBack13[[1,4],:]+0
        duplicatedBack13[[2,5],:]=duplicatedBack13[[2,5],:]+0
        duplicatedBack16[[0,3],:]=duplicatedBack16[[0,3],:]+L
        duplicatedBack16[[1,4],:]=duplicatedBack16[[1,4],:]-W
        duplicatedBack16[[2,5],:]=duplicatedBack16[[2,5],:]+0
        
        duplicatedBack19[[0,3],:]=duplicatedBack19[[0,3],:]+L
        duplicatedBack19[[1,4],:]=duplicatedBack19[[1,4],:]+W
        duplicatedBack19[[2,5],:]=duplicatedBack19[[2,5],:]-H
        duplicatedBack22[[0,3],:]=duplicatedBack22[[0,3],:]+L
        duplicatedBack22[[1,4],:]=duplicatedBack22[[1,4],:]+0
        duplicatedBack22[[2,5],:]=duplicatedBack22[[2,5],:]-H
        duplicatedBack25[[0,3],:]=duplicatedBack25[[0,3],:]+L
        duplicatedBack25[[1,4],:]=duplicatedBack25[[1,4],:]-W
        duplicatedBack25[[2,5],:]=duplicatedBack25[[2,5],:]-H
        
        #%%%% Front 
        duplicatedFront3=np.copy(duplicatedFront)
        duplicatedFront6=np.copy(duplicatedFront)
        duplicatedFront9=np.copy(duplicatedFront)
        duplicatedFront12=np.copy(duplicatedFront)
        duplicatedFront15=np.copy(duplicatedFront)
        duplicatedFront18=np.copy(duplicatedFront)
        duplicatedFront21=np.copy(duplicatedFront)
        duplicatedFront24=np.copy(duplicatedFront)
        duplicatedFront27=np.copy(duplicatedFront)
        
        duplicatedFront3[[0,3],:]=duplicatedFront3[[0,3],:]-L
        duplicatedFront3[[1,4],:]=duplicatedFront3[[1,4],:]+W
        duplicatedFront3[[2,5],:]=duplicatedFront3[[2,5],:]+H
        duplicatedFront6[[0,3],:]=duplicatedFront6[[0,3],:]-L
        duplicatedFront6[[1,4],:]=duplicatedFront6[[1,4],:]+0
        duplicatedFront6[[2,5],:]=duplicatedFront6[[2,5],:]+H
        duplicatedFront9[[0,3],:]=duplicatedFront9[[0,3],:]-L
        duplicatedFront9[[1,4],:]=duplicatedFront9[[1,4],:]-W
        duplicatedFront9[[2,5],:]=duplicatedFront9[[2,5],:]+H
        
        duplicatedFront12[[0,3],:]=duplicatedFront12[[0,3],:]-L
        duplicatedFront12[[1,4],:]=duplicatedFront12[[1,4],:]+W
        duplicatedFront12[[2,5],:]=duplicatedFront12[[2,5],:]+0
        duplicatedFront15[[0,3],:]=duplicatedFront15[[0,3],:]-L
        duplicatedFront15[[1,4],:]=duplicatedFront15[[1,4],:]+0
        duplicatedFront15[[2,5],:]=duplicatedFront15[[2,5],:]+0
        duplicatedFront18[[0,3],:]=duplicatedFront18[[0,3],:]-L
        duplicatedFront18[[1,4],:]=duplicatedFront18[[1,4],:]-W
        duplicatedFront18[[2,5],:]=duplicatedFront18[[2,5],:]+0
        
        duplicatedFront21[[0,3],:]=duplicatedFront21[[0,3],:]-L
        duplicatedFront21[[1,4],:]=duplicatedFront21[[1,4],:]+W
        duplicatedFront21[[2,5],:]=duplicatedFront21[[2,5],:]-H
        duplicatedFront24[[0,3],:]=duplicatedFront24[[0,3],:]-L
        duplicatedFront24[[1,4],:]=duplicatedFront24[[1,4],:]+0
        duplicatedFront24[[2,5],:]=duplicatedFront24[[2,5],:]-H
        duplicatedFront27[[0,3],:]=duplicatedFront27[[0,3],:]-L
        duplicatedFront27[[1,4],:]=duplicatedFront27[[1,4],:]-W
        duplicatedFront27[[2,5],:]=duplicatedFront27[[2,5],:]-H
        
        #%%%% Right
        duplicatedRight7=np.copy(duplicatedRight)
        duplicatedRight8=np.copy(duplicatedRight)
        duplicatedRight9=np.copy(duplicatedRight)
        duplicatedRight16=np.copy(duplicatedRight)
        duplicatedRight17=np.copy(duplicatedRight)
        duplicatedRight18=np.copy(duplicatedRight)
        duplicatedRight25=np.copy(duplicatedRight)
        duplicatedRight26=np.copy(duplicatedRight)
        duplicatedRight27=np.copy(duplicatedRight)
        
        duplicatedRight7[[0,3],:]=duplicatedRight7[[0,3],:]-L
        duplicatedRight7[[1,4],:]=duplicatedRight7[[1,4],:]+W
        duplicatedRight7[[2,5],:]=duplicatedRight7[[2,5],:]+H
        duplicatedRight8[[0,3],:]=duplicatedRight8[[0,3],:]+0
        duplicatedRight8[[1,4],:]=duplicatedRight8[[1,4],:]+W
        duplicatedRight8[[2,5],:]=duplicatedRight8[[2,5],:]+H
        duplicatedRight9[[0,3],:]=duplicatedRight9[[0,3],:]+L
        duplicatedRight9[[1,4],:]=duplicatedRight9[[1,4],:]+W
        duplicatedRight9[[2,5],:]=duplicatedRight9[[2,5],:]+H
        
        duplicatedRight16[[0,3],:]=duplicatedRight16[[0,3],:]-L
        duplicatedRight16[[1,4],:]=duplicatedRight16[[1,4],:]+W
        duplicatedRight16[[2,5],:]=duplicatedRight16[[2,5],:]+0
        duplicatedRight17[[0,3],:]=duplicatedRight17[[0,3],:]-0
        duplicatedRight17[[1,4],:]=duplicatedRight17[[1,4],:]+W
        duplicatedRight17[[2,5],:]=duplicatedRight17[[2,5],:]+0
        duplicatedRight18[[0,3],:]=duplicatedRight18[[0,3],:]+L
        duplicatedRight18[[1,4],:]=duplicatedRight18[[1,4],:]+W
        duplicatedRight18[[2,5],:]=duplicatedRight18[[2,5],:]+0
        
        duplicatedRight25[[0,3],:]=duplicatedRight25[[0,3],:]-L
        duplicatedRight25[[1,4],:]=duplicatedRight25[[1,4],:]+W
        duplicatedRight25[[2,5],:]=duplicatedRight25[[2,5],:]-H
        duplicatedRight26[[0,3],:]=duplicatedRight26[[0,3],:]+0
        duplicatedRight26[[1,4],:]=duplicatedRight26[[1,4],:]+W
        duplicatedRight26[[2,5],:]=duplicatedRight26[[2,5],:]-H
        duplicatedRight27[[0,3],:]=duplicatedRight27[[0,3],:]+L
        duplicatedRight27[[1,4],:]=duplicatedRight27[[1,4],:]+W
        duplicatedRight27[[2,5],:]=duplicatedRight27[[2,5],:]-H
        
        #%%%% Left
        duplicatedLeft1=np.copy(duplicatedLeft)
        duplicatedLeft2=np.copy(duplicatedLeft)
        duplicatedLeft3=np.copy(duplicatedLeft)
        duplicatedLeft10=np.copy(duplicatedLeft)
        duplicatedLeft11=np.copy(duplicatedLeft)
        duplicatedLeft12=np.copy(duplicatedLeft)
        duplicatedLeft19=np.copy(duplicatedLeft)
        duplicatedLeft20=np.copy(duplicatedLeft)
        duplicatedLeft21=np.copy(duplicatedLeft)
        
        duplicatedLeft1[[0,3],:]=duplicatedLeft1[[0,3],:]-L
        duplicatedLeft1[[1,4],:]=duplicatedLeft1[[1,4],:]-W
        duplicatedLeft1[[2,5],:]=duplicatedLeft1[[2,5],:]+H
        duplicatedLeft2[[0,3],:]=duplicatedLeft2[[0,3],:]+0
        duplicatedLeft2[[1,4],:]=duplicatedLeft2[[1,4],:]-W
        duplicatedLeft2[[2,5],:]=duplicatedLeft2[[2,5],:]+H
        duplicatedLeft3[[0,3],:]=duplicatedLeft3[[0,3],:]+L
        duplicatedLeft3[[1,4],:]=duplicatedLeft3[[1,4],:]-W
        duplicatedLeft3[[2,5],:]=duplicatedLeft3[[2,5],:]+H
        
        duplicatedLeft10[[0,3],:]=duplicatedLeft10[[0,3],:]-L
        duplicatedLeft10[[1,4],:]=duplicatedLeft10[[1,4],:]-W
        duplicatedLeft10[[2,5],:]=duplicatedLeft10[[2,5],:]+0
        duplicatedLeft11[[0,3],:]=duplicatedLeft11[[0,3],:]-0
        duplicatedLeft11[[1,4],:]=duplicatedLeft11[[1,4],:]-W
        duplicatedLeft11[[2,5],:]=duplicatedLeft11[[2,5],:]+0
        duplicatedLeft12[[0,3],:]=duplicatedLeft12[[0,3],:]+L
        duplicatedLeft12[[1,4],:]=duplicatedLeft12[[1,4],:]-W
        duplicatedLeft12[[2,5],:]=duplicatedLeft12[[2,5],:]+0
        
        duplicatedLeft19[[0,3],:]=duplicatedLeft19[[0,3],:]-L
        duplicatedLeft19[[1,4],:]=duplicatedLeft19[[1,4],:]-W
        duplicatedLeft19[[2,5],:]=duplicatedLeft19[[2,5],:]-H
        duplicatedLeft20[[0,3],:]=duplicatedLeft20[[0,3],:]+0
        duplicatedLeft20[[1,4],:]=duplicatedLeft20[[1,4],:]-W
        duplicatedLeft20[[2,5],:]=duplicatedLeft20[[2,5],:]-H
        duplicatedLeft21[[0,3],:]=duplicatedLeft21[[0,3],:]+L
        duplicatedLeft21[[1,4],:]=duplicatedLeft21[[1,4],:]-W
        duplicatedLeft21[[2,5],:]=duplicatedLeft21[[2,5],:]-H
        
        #%%%% Top
        duplicatedTop1=np.copy(duplicatedTop)
        duplicatedTop2=np.copy(duplicatedTop)
        duplicatedTop3=np.copy(duplicatedTop)
        duplicatedTop4=np.copy(duplicatedTop)
        duplicatedTop5=np.copy(duplicatedTop)
        duplicatedTop6=np.copy(duplicatedTop)
        duplicatedTop7=np.copy(duplicatedTop)
        duplicatedTop8=np.copy(duplicatedTop)
        duplicatedTop9=np.copy(duplicatedTop)
        
        duplicatedTop1[[0,3],:]=duplicatedTop1[[0,3],:]-L
        duplicatedTop1[[1,4],:]=duplicatedTop1[[1,4],:]+W
        duplicatedTop1[[2,5],:]=duplicatedTop1[[2,5],:]-H
        duplicatedTop2[[0,3],:]=duplicatedTop2[[0,3],:]+0
        duplicatedTop2[[1,4],:]=duplicatedTop2[[1,4],:]+W
        duplicatedTop2[[2,5],:]=duplicatedTop2[[2,5],:]-H
        duplicatedTop3[[0,3],:]=duplicatedTop3[[0,3],:]+L
        duplicatedTop3[[1,4],:]=duplicatedTop3[[1,4],:]+W
        duplicatedTop3[[2,5],:]=duplicatedTop3[[2,5],:]-H
        
        duplicatedTop4[[0,3],:]=duplicatedTop4[[0,3],:]-L
        duplicatedTop4[[1,4],:]=duplicatedTop4[[1,4],:]+0
        duplicatedTop4[[2,5],:]=duplicatedTop4[[2,5],:]-H
        duplicatedTop5[[0,3],:]=duplicatedTop5[[0,3],:]+0
        duplicatedTop5[[1,4],:]=duplicatedTop5[[1,4],:]+0
        duplicatedTop5[[2,5],:]=duplicatedTop5[[2,5],:]-H
        duplicatedTop6[[0,3],:]=duplicatedTop6[[0,3],:]+L
        duplicatedTop6[[1,4],:]=duplicatedTop6[[1,4],:]+0
        duplicatedTop6[[2,5],:]=duplicatedTop6[[2,5],:]-H
        
        duplicatedTop7[[0,3],:]=duplicatedTop7[[0,3],:]-L
        duplicatedTop7[[1,4],:]=duplicatedTop7[[1,4],:]-W
        duplicatedTop7[[2,5],:]=duplicatedTop7[[2,5],:]-H
        duplicatedTop8[[0,3],:]=duplicatedTop8[[0,3],:]+0
        duplicatedTop8[[1,4],:]=duplicatedTop8[[1,4],:]-W
        duplicatedTop8[[2,5],:]=duplicatedTop8[[2,5],:]-H
        duplicatedTop9[[0,3],:]=duplicatedTop9[[0,3],:]+L
        duplicatedTop9[[1,4],:]=duplicatedTop9[[1,4],:]-W
        duplicatedTop9[[2,5],:]=duplicatedTop9[[2,5],:]-H
        
        #%%%% Bottom
        duplicatedBottom19=np.copy(duplicatedBottom)
        duplicatedBottom20=np.copy(duplicatedBottom)
        duplicatedBottom21=np.copy(duplicatedBottom)
        duplicatedBottom22=np.copy(duplicatedBottom)
        duplicatedBottom23=np.copy(duplicatedBottom)
        duplicatedBottom24=np.copy(duplicatedBottom)
        duplicatedBottom25=np.copy(duplicatedBottom)
        duplicatedBottom26=np.copy(duplicatedBottom)
        duplicatedBottom27=np.copy(duplicatedBottom)
        
        duplicatedBottom19[[0,3],:]=duplicatedBottom19[[0,3],:]-L
        duplicatedBottom19[[1,4],:]=duplicatedBottom19[[1,4],:]+W
        duplicatedBottom19[[2,5],:]=duplicatedBottom19[[2,5],:]+H
        duplicatedBottom20[[0,3],:]=duplicatedBottom20[[0,3],:]+0
        duplicatedBottom20[[1,4],:]=duplicatedBottom20[[1,4],:]+W
        duplicatedBottom20[[2,5],:]=duplicatedBottom20[[2,5],:]+H
        duplicatedBottom21[[0,3],:]=duplicatedBottom21[[0,3],:]+L
        duplicatedBottom21[[1,4],:]=duplicatedBottom21[[1,4],:]+W
        duplicatedBottom21[[2,5],:]=duplicatedBottom21[[2,5],:]+H
        
        duplicatedBottom22[[0,3],:]=duplicatedBottom22[[0,3],:]-L
        duplicatedBottom22[[1,4],:]=duplicatedBottom22[[1,4],:]+0
        duplicatedBottom22[[2,5],:]=duplicatedBottom22[[2,5],:]+H
        duplicatedBottom23[[0,3],:]=duplicatedBottom23[[0,3],:]+0
        duplicatedBottom23[[1,4],:]=duplicatedBottom23[[1,4],:]+0
        duplicatedBottom23[[2,5],:]=duplicatedBottom23[[2,5],:]+H
        duplicatedBottom24[[0,3],:]=duplicatedBottom24[[0,3],:]+L
        duplicatedBottom24[[1,4],:]=duplicatedBottom24[[1,4],:]+0
        duplicatedBottom24[[2,5],:]=duplicatedBottom24[[2,5],:]+H
        
        duplicatedBottom25[[0,3],:]=duplicatedBottom25[[0,3],:]-L
        duplicatedBottom25[[1,4],:]=duplicatedBottom25[[1,4],:]-W
        duplicatedBottom25[[2,5],:]=duplicatedBottom25[[2,5],:]+H
        duplicatedBottom26[[0,3],:]=duplicatedBottom26[[0,3],:]+0
        duplicatedBottom26[[1,4],:]=duplicatedBottom26[[1,4],:]-W
        duplicatedBottom26[[2,5],:]=duplicatedBottom26[[2,5],:]+H
        duplicatedBottom27[[0,3],:]=duplicatedBottom27[[0,3],:]+L
        duplicatedBottom27[[1,4],:]=duplicatedBottom27[[1,4],:]-W
        duplicatedBottom27[[2,5],:]=duplicatedBottom27[[2,5],:]+H    
        
        
        #%%% assign boundary lables
        #%%%% Back
        labelStart=1
        labelEnd=labelStart+duplicatedBack.shape[1]
        labelBack=np.arange(labelStart,labelEnd,1)
        fibrePosition[6,:]=labelBack
        duplicatedBack1[6,:]=labelBack
        duplicatedBack4[6,:]=labelBack
        duplicatedBack7[6,:]=labelBack
        duplicatedBack10[6,:]=labelBack
        duplicatedBack13[6,:]=labelBack
        duplicatedBack16[6,:]=labelBack
        duplicatedBack19[6,:]=labelBack
        duplicatedBack22[6,:]=labelBack
        duplicatedBack25[6,:]=labelBack
        
        
        #%%%% Front
        labelStart=labelEnd
        labelEnd=labelStart+duplicatedFront.shape[1]
        labelFront=np.arange(labelStart,labelEnd,1)
        fibrePosition[7,:]=labelFront
        duplicatedFront3[7,:]=labelFront
        duplicatedFront6[7,:]=labelFront
        duplicatedFront9[7,:]=labelFront
        duplicatedFront12[7,:]=labelFront
        duplicatedFront15[7,:]=labelFront
        duplicatedFront18[7,:]=labelFront
        duplicatedFront21[7,:]=labelFront
        duplicatedFront24[7,:]=labelFront
        duplicatedFront27[7,:]=labelFront
        
        #%%%% Right
        labelStart=labelEnd
        labelEnd=labelStart+duplicatedRight.shape[1]
        labelRight=np.arange(labelStart,labelEnd,1)
        fibrePosition[8,:]=labelRight
        duplicatedRight7[8,:]=labelRight
        duplicatedRight8[8,:]=labelRight
        duplicatedRight9[8,:]=labelRight
        duplicatedRight16[8,:]=labelRight
        duplicatedRight17[8,:]=labelRight
        duplicatedRight18[8,:]=labelRight
        duplicatedRight25[8,:]=labelRight
        duplicatedRight26[8,:]=labelRight
        duplicatedRight27[8,:]=labelRight
        
        #%%%% Left
        labelStart=labelEnd
        labelEnd=labelStart+duplicatedLeft.shape[1]
        labelLeft=np.arange(labelStart,labelEnd,1)
        fibrePosition[9,:]=labelLeft
        duplicatedLeft1[9,:]=labelLeft
        duplicatedLeft2[9,:]=labelLeft
        duplicatedLeft3[9,:]=labelLeft
        duplicatedLeft10[9,:]=labelLeft
        duplicatedLeft11[9,:]=labelLeft
        duplicatedLeft12[9,:]=labelLeft
        duplicatedLeft19[9,:]=labelLeft
        duplicatedLeft20[9,:]=labelLeft
        duplicatedLeft21[9,:]=labelLeft
        
        #%%%% Top
        labelStart=labelEnd
        labelEnd=labelStart+duplicatedTop.shape[1]
        labelTop=np.arange(labelStart,labelEnd,1)
        fibrePosition[10,:]=labelTop
        duplicatedTop1[10,:]=labelTop
        duplicatedTop2[10,:]=labelTop
        duplicatedTop3[10,:]=labelTop
        duplicatedTop4[10,:]=labelTop
        duplicatedTop5[10,:]=labelTop
        duplicatedTop6[10,:]=labelTop
        duplicatedTop7[10,:]=labelTop
        duplicatedTop8[10,:]=labelTop
        duplicatedTop9[10,:]=labelTop
        
        #%%%% Bottom
        labelStart=labelEnd
        labelEnd=labelStart+duplicatedBottom.shape[1]
        labelBottom=np.arange(labelStart,labelEnd,1)
        fibrePosition[11,:]=labelBottom
        duplicatedBottom19[11,:]=labelBottom
        duplicatedBottom20[11,:]=labelBottom
        duplicatedBottom21[11,:]=labelBottom
        duplicatedBottom22[11,:]=labelBottom
        duplicatedBottom23[11,:]=labelBottom
        duplicatedBottom24[11,:]=labelBottom
        duplicatedBottom25[11,:]=labelBottom
        duplicatedBottom26[11,:]=labelBottom
        duplicatedBottom27[11,:]=labelBottom
        
        
        #%%% add boundary set to fibrePosition
        fibrePositionFull=np.copy(fibrePosition)
        fibrePositionDuplicated=np.concatenate((duplicatedBack1,duplicatedBack4,duplicatedBack7,duplicatedBack10,duplicatedBack13,duplicatedBack16,duplicatedBack19,duplicatedBack22,duplicatedBack25,duplicatedFront3,duplicatedFront6,duplicatedFront9,duplicatedFront12,duplicatedFront15,duplicatedFront18,duplicatedFront21,duplicatedFront24,duplicatedFront27,duplicatedRight7,duplicatedRight8,duplicatedRight9,duplicatedRight16,duplicatedRight17,duplicatedRight18,duplicatedRight25,duplicatedRight26,duplicatedRight27,duplicatedLeft1,duplicatedLeft2,duplicatedLeft3,duplicatedLeft10,duplicatedLeft11,duplicatedLeft12,duplicatedLeft19,duplicatedLeft20,duplicatedLeft21,duplicatedTop1,duplicatedTop2,duplicatedTop3,duplicatedTop4,duplicatedTop5,duplicatedTop6,duplicatedTop7,duplicatedTop8,duplicatedTop9,duplicatedBottom19,duplicatedBottom20,duplicatedBottom21,duplicatedBottom22,duplicatedBottom23,duplicatedBottom24,duplicatedBottom25,duplicatedBottom26,duplicatedBottom27),axis=1)
        
        #%%%% remove duplicated boundary fibres 
        dummyOutput, uniqueFibreIndex=np.unique(fibrePositionDuplicated[0:6,:], axis=1, return_index=True)
        fibrePositionDuplicated=fibrePositionDuplicated[:,uniqueFibreIndex]
        fibrePositionFull=np.concatenate((fibrePositionFull,fibrePositionDuplicated),axis=1)
        #%% output fibre layout
        
        return fibrePositionFull
