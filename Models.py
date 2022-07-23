from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from Layers import *
from utils import *

def Binary(y_true,y_pred):
    #return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)


def Discriminator(source_shape=(None,None,5), target_shape=(None,None,1), feature_shape = (None,54)):

	# source image input
    in_src_image = Input(shape= source_shape)
    # 1D feature inputs
    in1Dfeatures = Input(shape=feature_shape)
    l1 = Outer_Dot(in1Dfeatures)

	# target image input
    in_target_image = Input(shape= target_shape)
	# concatenate images channel-wise
    merged = Concat([in_src_image,l1, in_target_image])
	# C64
    d = Conv(merged, 64, (4,4),strides=(1, 1), padding='same')
	# C128
    d = Conv(d, 128, (4,4), strides=(2,2), padding='same')
	# C256
    d = Conv(d, 256, (4,4), strides=(2,2), padding='same')

	# C512
    d = Conv(d, 512, (4,4), strides=(2,2), padding='same')
	# patch output
    patch_out = Dis_classifier(d)
	# define model
    model = Model([in_src_image, in_target_image], patch_out)
	# compile model
    opt = Adam(lr=0.0004, beta_1=0.5)
    #bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=Binary, optimizer=opt, loss_weights=[0.5])
    return model



def Generator(args, lat_shape =(None,None,1), image_shape=(None,None,5), feature_shape = (None,54) ):
    lat_space = Input(shape = lat_shape)

    in1D_feutures = Input(shape = feature_shape)
    l1 = Outer_Dot_gen(in1D_feutures)
    in_image = Input(shape=image_shape)
    x = Conv(in_image, filters = 32, kernel_size = 2, strides = 1,padding='valid', batch = True, active = 'reLU')

    y = SE_Concat(64, x)
    for _ in range(args.SE_concat - 1):
        y = SE_Concat(64, y)

    y = Conv(y, filters = 64, kernel_size = 3, strides = 2,padding='same', batch = True, active = 'reLU')

    y1 = SE_Concat(128, y)
    for _ in range(args.SE_concat - 1):
        y1 = SE_Concat(128, y1)
    y1 = Conv(y1, filters = 128, kernel_size = 3, strides = 2,padding='same', batch = True, active = 'reLU')


    y2 = SE_Concat(256, y1)
    for _ in range(args.SE_concat - 1):
        y2 = SE_Concat(256, y2)
    y2 = Conv(y2, filters = 128, kernel_size = 3, strides = 2,padding='same', batch = True, active = 'reLU')

    y2 = SE_Concat(256, y2)
    for _ in range(args.SE_concat - 1):
        y2 = SE_Concat(256, y2)


    g = Concatenate()([lat_space, l1])


    a = synthesis_block(g, y2,512)
    a = upsampling(a, 256)

    a = synthesis_block(a, y1,128)
    a = upsampling(a, 128)

    a = synthesis_block(a, y,64)
    a = upsampling(a, 64)

    out_image = Conv(a, filters = 1, kernel_size = 7, strides = 1,padding='same', batch = True, active = 'sigmoid')

    # define model
    model = Model([in_image, in1D_feutures, lat_space], out_image)
    return model





def GAN(g_model, d_model):
	# make weights in the discriminator not trainable
    d_model.trainable = False
	# define the source image
    lat_input = Input(shape=(None,None,1))
    in1D_features = Input(shape=(None,54))
    in_src = Input(shape=(None,None,5))
	# connect the source image to the generator input
    gen_out = g_model([in_src,in1D_features, lat_input])
	# connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src,in1D_features, gen_out])
	# src image as input, generated image and classification output
    model = Model([in_src,in1D_features,lat_input], [dis_out, gen_out])
	# compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=[bce, 'bce'], optimizer=opt)
    return model



