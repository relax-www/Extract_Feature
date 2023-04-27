from scipy import interpolate
import numpy as np
def cubic3(X,lag_same,loc_l,loc_r):
    chuli=X[:,loc_l:loc_r+1]
    sample_size=chuli.shape[0]
    feature_size=chuli.shape[1]
    for i in range(0,sample_size-sample_size%lag_same,lag_same):
        if i==0:
            temp=np.reshape(chuli[i,:],[1,feature_size])
        else:
            temp=np.append(temp,np.reshape(chuli[i,:],[1,feature_size]),axis=0)
    y=np.linspace(0, int(X.shape[0]/lag_same), num=int(X.shape[0]/lag_same))
    y_larger=np.linspace(0, int(X.shape[0]/lag_same), num=sample_size)

    for i in range(0,feature_size):
        f2 = interpolate.interp1d(y,temp[:,i],kind='linear')
        result=f2(y_larger).reshape(sample_size,1)
        if i==0:
            x_pred = result
        else:
            x_pred=np.append(x_pred,result,axis=1)
    X[:,loc_l:loc_r+1]=x_pred
    return X