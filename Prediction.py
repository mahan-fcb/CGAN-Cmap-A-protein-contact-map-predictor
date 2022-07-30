import numpy as np
from tensorflow.keras.models import load_model
from evaluation import evaluate
import tensorflow.keras as keras
from utils import *
def myloss(ytrue,ypred):
    mae = keras.losses.MeanAbsoluteError()
    bce = keras.losses.BinaryCrossentropy()
    return bce(ytrue,ypred) 







# model = load_model('./g_model_101_NET5.h5',
#                    custom_objects={'myloss':myloss})

def prediction(model,feature_3d, feature_1d):
    lat = generate_latent_points(np.size(feature_3d,1), np.size(feature_3d,0))
    pred = model.predict([feature_3d, feature_1d,lat])
    pred = pred.reshape(np.size(pred,0),np.size(pred,1),np.size(pred,2))
    for i in range(np.size(pred,0)):
        #pred[i,:,:] = np.maximum( pred[i,:,:], pred[i,:,:].transpose() )
        pred[i,:,:] = 0.5*(pred[i,:,:]+pred[i,:,:].transpose())
    #     pred = np.where(pred>1,1,pred)
    # pred = np.round(pred)
    return pred

from matplotlib import pyplot
def Save_predicted_images(cfg, pred,gtruth):
    for i in range(len(pred)):
        pyplot.subplot(3, len(pred), 1 + len(pred) + i)
        pyplot.axis('off')
        pyplot.imshow(pred[i])
	# plot real target image
    for i in range(len(pred)):
        pyplot.subplot(3, len(pred), 1 + len(pred)*2 + i)
        pyplot.axis('off')
        pyplot.imshow(gtruth[i])
	# save plot to file
    filename1 = cfg.pred_dir + '/Prediction_'+ cdf.test_data+'_testdata.png'
    pyplot.savefig(filename1)
    pyplot.close()
	# save the generator model
 



def Predict(cfg,model, test_dataset):
    f_3dt, f_1dt, co_mapt = test_dataset
    n_samples = 1
    bat_test = int(len(f_3dt) / n_samples)
    L2 = []
    ACC1 = []
    ACC2=  []
    predlist = []
    gtruthlist = []
    for i in range(bat_test):
        f3t, f1t, cmt = generate_real_samples(test_dataset, n_samples, i)
        cmt = cmt.reshape(np.size(cmt,1),np.size(cmt,2))


        pred = prediction(model,f3t, f1t)
        pred = pred.reshape(np.size(pred,1),np.size(pred,2))
        for jjj in range(cmt.shape[0]):
            if (1 - cmt[jjj,:]).all() and (1 - cmt[:,jjj]).all():
                pred[jjj,:] = 0 
                pred[:,jjj] = 0
        acc1,acc2 = evaluate(pred,cmt)
        ACC1.append(acc1)
        ACC2.append(acc2)
        if i % 80 == 0:
            predlist.append(pred)
            gtruthlist.append(cmt)

    Save_predicted_images(cfg, predlist,gtruthlist)
    #f = 'Accuracy'
    #f1 = 'Recall'
    #np.save(f,ACC1)
    #np.save(f1,ACC2)

    long = np.array(ACC1)[0,1,0]
    for i in range(1,len(ACC1)):
        long = long + ACC1[i,1,0]
    
    l2 = long/len(ACC1)
    L2.append(l2)
    #np.save('l2',L2)

    print('>Accuracy[%f], Recall[%f], L2[%f]' % (ACC1,ACC2, L2))
    return np.array(ACC1), np.array(ACC2) np.array(L2)

