from matplotlib import pyplot
import numpy as np
import h5py
import os
from numpy.random import randn
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)
        
        
def Delete_Nan(xtrain,ytrain):
    y = np.argwhere(np.isnan(ytrain))
    y = y[:,0]
    y = np.unique(y)
    for i in range(np.size(y)-1,-1,-1):
        ytrain = np.delete(ytrain, y[i], 0)
        xtrain = np.delete(xtrain, y[i], 0)
        
    return xtrain,ytrain



# load and prepare training images
def load_real_samples(cfg, train = True):
    if train:
        feature_3d = np.load(cfg.data_root+'/Train_Set/LxLx5_sort_train.npy',allow_pickle=True)
        feature_1d = np.load(cfg.data_root+'/Train_Set/Lx54_sort_train.npy',allow_pickle=True)
        contact_map = np.load(cfg.data_root+'/Train_Set/contact_sort_train.npy',allow_pickle=True)
    else:
        feature_3d = np.load(cfg.feature_3d,allow_pickle=True)
        feature_1d = np.load(cfg.feature_1d,allow_pickle=True)
        contact_map = np.load(cfg.contact_map,allow_pickle=True)
    
    
    return [feature_3d, feature_1d, contact_map]

 
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, i):
	# unpack dataset
    feature_3d, feature_1d, contact_map = dataset
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
    y = 0.9*ones((n_samples, 1, 1, 1))
    return [final_X1, final_X2, final_X3], y


    
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(int(latent_dim/8) * n_samples* int(latent_dim/8))
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, int(latent_dim/8), int(latent_dim/8),1)
	return x_input



# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples_3d, samples_1d):
	# generate fake instance
    lat_space = generate_latent_points(np.size(samples_3d,1), np.size(samples_3d,0))
    X = g_model.predict([samples_3d, samples_1d,lat_space])
	# create 'fake' class labels (0)
    y = 0.1*ones((len(X), 1, 1, 1))
    return X, y



# generate samples and save as a plot and save the model
def summarize_performance(step,g_model,d_model, dataset, bat,model_dir,log_dir, n_samples=4):
    a = int(randint(bat, size=1)[0])
	# select a sample of input images
    [feature_3d, feature_1d, contact_map], y_real = generate_real_samples(dataset, n_samples,a)
	#[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model,feature_3d, feature_1d)
	#X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i,:,:,0])
	# plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(contact_map[i,:,:,0])
	# save plot to file
    filename1 = log_dir+'/plot_%03d_GAN3.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
	# save the generator model
    filename2 = model_dir+'/g_model_%03d_GAN3.h5' % (step+1)
    g_model.save(filename2)
    
    filename3 = model_dir+'/d_model_%03d_GAN3.h5' % (step+1)
    d_model.save(filename3)
    print('>Saved: %s and %s and %s' % (filename1, filename2, filename3))


       
def save_prediction(n_testsamples, pred_dir, ytest,ypred):
    for i in range((n_testsamples if n_testsamples<ypred.shape[0] else ypred.shape[0])):
        pyplot.figure(figsize=(10, 10))
        pyplot.axis('off')
        pyplot.imshow(ytest[i,:,:,0])
        filename1 = pred_dir+'\\gt_%d.png' % (i)
        pyplot.savefig(filename1)
        pyplot.close()
        
        pyplot.figure(figsize=(10, 10))
        pyplot.axis('off')
        pyplot.imshow(ypred[i,:,:,0])
        filename2 = pred_dir+'\\yp_%d.png' % (i)
        pyplot.savefig(filename2)
        pyplot.close()
    
    
def Load_pretrain_model(model_dir):
    base_model = load_model(model_dir)
    mmodel = Model(inputs=base_model.input,outputs= base_model.layers[-2].output)
    mmodel.trainable = True
    return mmodel


    
