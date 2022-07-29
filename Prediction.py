import numpy as np
from tensorflow.keras.models import load_model
from evaluation import evaluate
import tensorflow.keras as keras

def myloss(ytrue,ypred):
    mae = keras.losses.MeanAbsoluteError()
    bce = keras.losses.BinaryCrossentropy()
    return bce(ytrue,ypred) 


def load_test_samples():
    
    feature_3d = np.load('./f3-final-casp13.npy',allow_pickle=True)
    feature_1d = np.load('./f1-final-casp13.npy',allow_pickle=True)
    contact_map = np.load('./co-final-casp13.npy',allow_pickle=True)
    
    
    return [feature_3d, feature_1d, contact_map] 
    
 
    
def generate_real_samples(feature_3d, feature_1d, contact_map, n_samples, i):
	# unpack dataset
    #feature_3d, feature_1d, contact_map = dataset
	# retrieve selected images
    X1, X2, X3 = feature_3d[i*n_samples : (1+i)*n_samples], feature_1d[i*n_samples : (1+i)*n_samples],contact_map[i*n_samples : (1+i)*n_samples]
	# generate 'real' class labels (1)
    pad_width = 8 - X1[n_samples-1].shape[0]%8
    final_X1 = []
    final_X2 = []
    final_X3 = []
    for ii in range(n_samples):
        pad = pad_width + X1[n_samples-1].shape[0] - X1[ii].shape[0]
        final_X1.append(np.pad(X1[ii], ((0,pad),(0,pad),(0,0)), 'constant', constant_values=0))
        final_X2.append(np.pad(X2[ii], ((0,pad),(0,0)), 'constant', constant_values=0))
        
        a = np.where(X3[ii] == 1, X3[ii], 0)
        final_X3.append(np.pad(a, ((0,pad),(0,pad)), 'constant', constant_values=0))
        
    final_X1 = np.array(final_X1).astype(np.float32)
    final_X2 = np.array(final_X2).astype(np.float32)
    final_X3 = np.array(final_X3).astype(np.float32)
    final_X3 = final_X3.reshape((final_X3.shape[0],final_X3.shape[1] ,final_X3.shape[2], 1))
    return [final_X1, final_X2, final_X3]




from numpy.random import randn
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(int(latent_dim/8) * n_samples* int(latent_dim/8))
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, int(latent_dim/8), int(latent_dim/8),1)
	return x_input


# model = load_model('./g_model_101_NET5.h5',
#                    custom_objects={'myloss':myloss})

def prediction(feature_3d, feature_1d):
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
def summarize_performance(pred,gtruth):
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
    filename1 = 'plot1.png'
    pyplot.savefig(filename1)
    pyplot.close()
	# save the generator model
 




f_3dt, f_1dt, co_mapt = load_test_samples()
n_samples = 1
bat_test = int(len(f_3dt) / n_samples)
L2 = []
for ii in range(240,251,10):
    file_name = './g_model_%03d_GAN3.h5' % (ii+1) 
    model = load_model(file_name)
    ACC1 = []
    ACC2=  []
    predlist = []
    gtruthlist = []
    for i in range(bat_test):
        f3t, f1t, cmt = generate_real_samples(f_3dt, f_1dt, co_mapt, n_samples, i)
        cmt = cmt.reshape(np.size(cmt,1),np.size(cmt,2))


        pred = prediction(f3t, f1t)
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

    summarize_performance(predlist,gtruthlist)
    f = 'Accuracy_%03d_GAN3_resumed' % (ii+1)
    f1 = 'Recall_%03d_GAN3_resumed' % (ii+1)
    np.save(f,ACC1)
    np.save(f1,ACC2)
L2 = []
for ii in range(0,371,10):
    file_name = 'Accuracy_%03d_GAN3_resumed.npy' % (ii+1)
    acc = np.load(file_name)
    long = acc[0,1,0]
    for i in range(1,len(acc)):
        long = long + acc[i,1,0]
    
    l2 = long/len(acc)
    L2.append(l2)
np.save('l2.npy',L2)