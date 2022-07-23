from Models import *
from utils import *


# train pix2pix models
def train(cfg, d_model, g_model, gan_model, dataset):
	# determine the output square shape of the discriminator
    #n_patch = d_model.output_shape[1]
    #n_patch = int(np.floor(dataset[1].shape[1]/4))+1
	# unpack dataset
    f_3d, f_1d, co_map = dataset
	# calculate the number of batches per training epoch
    bat_per_epo = int(len(f_3d) / cfg.batch_size)
	# calculate the number of training iterations
    #n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
    for j in range(cfg.n_epoch):

        for i in range(bat_per_epo):
    		# select a batch of real samples
            [feature_3d, feature_1d, contact_map], y_real = generate_real_samples(dataset, cfg.batch_size,i)
            lat_space = generate_latent_points(np.size(feature_3d,1), np.size(feature_3d,0))
    		# generate a batch of fake samples
            contact_map_fake, y_fake = generate_fake_samples(g_model,feature_3d, feature_1d)
    		# update discriminator for real samples
            d_loss1 = d_model.train_on_batch([feature_3d, feature_1d, contact_map], y_real)
    		# update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([feature_3d, feature_1d, contact_map_fake], y_fake)
    		# update the generator
            g_loss, _, _ = gan_model.train_on_batch([feature_3d, feature_1d,lat_space], [y_real, contact_map])
    		# summarize performance
            print('>Epoch[%d], Iter[%d], d1[%.3f] d2[%.3f] g[%.3f]' % (j+1,i+1, d_loss1, d_loss2, g_loss))
    		# summarize model performance
        if j % cfg.save_step == 0:
            summarize_performance(j, g_model,d_model, dataset,bat_per_epo, cfg.model_dir, cfg.log_dir)



