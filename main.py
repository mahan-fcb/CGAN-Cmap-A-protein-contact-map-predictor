from Models import *
from config import * 
from trainer import * 
from utils import * 
from tensorflow.keras.models import load_model
from prediction import *
cfg = ConfigGAN()

if cfg.traintest == 'traintest':
    
    dataset = load_real_samples(cfg, train = True)
    print('Dataset is loaded')
    d_model = Discriminator()
    g_model = Generator(cfg)
    # define the composite model
    gan_model = GAN(g_model, d_model)
    # train model
    model = train(cfg,d_model, g_model, gan_model, dataset)


    test_dataset = load_real_samples(cfg, train  = False)
    Accuracy, Recall, L2 = Predict(cfg,model, test_dataset)
    
    
if cfg.traintest == 'test':
   test_dataset = load_real_samples(cfg, train  = False)

   model = load_model(cfg.pretrain_model)
   Accuracy, Recall, L2 = Predict(cfg,model, test_dataset)
 #   ypred = model.predict([test_dataset[0],lat_input])
    
 #   save_prediction(cfg.n_testsamples, cfg.pred_dir, test_dataset[1],ypred)
 #   print('The predicted images has been saved!')
