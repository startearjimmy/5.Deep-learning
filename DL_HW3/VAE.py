from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array

learning_rate = 0.001
batch_size = 256
n_latent = 100

### (i) Preprocessing training data
path = "cartoon/"
listing_train = os.listdir(path)
image_list_train = []
#
for image in listing_train:                     
    im = Image.open(path + image)
    im = im.convert('RGB')
    im_res = im.resize((32,32), Image.BILINEAR)   
    image_list_train.append(array(im_res)/255.0)
        
image_list_train = array(image_list_train,dtype='float32')        
train_x = image_list_train.reshape(-1,32,32,3)
tf.reset_default_graph()

X_in = tf.placeholder(dtype=tf.float32, shape=[None,32,32,3], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None,32,32,3], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 32*32*3])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1,32,32,3])
        x = tf.layers.conv2d(X, filters=128, kernel_size=3, strides=2, padding='same', activation=activation) # 16x16x128 
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same', activation=activation) # 8x8x256
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', activation=activation) # 4x4x512
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x) # 4x4x512 = 8192
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))        
        return z, mn, sd

def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=4*4*512, activation=lrelu)
        x = tf.reshape(x, shape=[-1,4,4,512])
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu) # 8x8x256
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu) # 16x16x128
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu) # 32x32x3        
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=32*32*3, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1,32,32,3])
        return img

sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, shape=[-1,32*32*3])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost = []
for i in range(151):
    for batch in range(len(train_x)//batch_size):
        batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
        sess.run(optimizer, feed_dict = {X_in: batch_x, Y: batch_x, keep_prob: 0.8})
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch_x, Y: batch_x, keep_prob: 1.0})
    plt.imshow(batch_x[0])
    plt.show()
    plt.imshow(d[0])
    plt.show()
    print("Iter " + str(i) + ", Loss= " + \
          "{:.4f}".format(ls) + ", Image Loss = " + \
          "{:.4f}".format(np.mean(i_ls)) + ", Latent Loss = " + \
          "{:.4f}".format(np.mean(d_ls)))    
    cost.append(ls)

## (i) plot learning curve
epoch = np.linspace(0,150,16)
plt.plot(range(len(cost)), cost, 'b')
plt.title('Learning Curve of VAE')
plt.xlabel('Epoch',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.xticks(epoch)
plt.legend()
plt.figure()
plt.show()

## (ii) Show some examples reconstructed by your model.
eachSize=32 
eachLine=8

Real = Image.new('RGB', (eachSize*eachLine,eachSize*eachLine))
x = 0
y = 0
for i in range(len(batch_x)):
    d_img = Image.fromarray(np.uint8(batch_x[i]*255.0),"RGB")
    d_img = d_img.resize((eachSize, eachSize), Image.ANTIALIAS)
    Real.paste(d_img, (x * eachSize, y * eachSize))
    x += 1
    if x == eachLine:
        x = 0
        y += 1
plt.imshow(Real)
plt.title("Real samples in dataset")
plt.show()

Reconstruction = Image.new('RGB', (eachSize*eachLine,eachSize*eachLine))
x = 0
y = 0
for i in range(len(d)):
    d_img = Image.fromarray(np.uint8(d[i]*255.0),"RGB")
    d_img = d_img.resize((eachSize, eachSize), Image.ANTIALIAS)
    Reconstruction.paste(d_img, (x * eachSize, y * eachSize))
    x += 1
    if x == eachLine:
        x = 0
        y += 1
plt.imshow(Reconstruction)
plt.title("Reconstruction samples using VAE")
plt.show()

## (iii) Sample the prior p(z) to generate some examples when your model is well-trained.
randoms = [np.random.normal(0, 1, n_latent) for _ in range(100)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
Random = Image.new('RGB', (eachSize*eachLine,eachSize*eachLine))
x = 0
y = 0
for i in range(len(imgs)):
    d_img = Image.fromarray(np.uint8(imgs[i]*255.0),"RGB")
    d_img = d_img.resize((eachSize, eachSize), Image.ANTIALIAS)
    Random.paste(d_img, (x * eachSize, y * eachSize))
    x += 1
    if x == eachLine:
        x = 0
        y += 1
plt.imshow(Random)
plt.title("Samples drawn from VAE")
plt.show()

eachSize=32 
eachLine=4

Real = Image.new('RGB', (eachSize*eachLine*2,eachSize*eachLine*2))
x = 0
y = 0
for i in range(len(batch_x)):
    d_img = Image.fromarray(np.uint8(batch_x[i]*255.0),"RGB")
    d_img = d_img.resize((eachSize, eachSize), Image.ANTIALIAS)
    Real.paste(d_img, (x * eachSize, y * eachSize))
    x += 1
    d_img = Image.fromarray(np.uint8(d[i]*255.0),"RGB")
    d_img = d_img.resize((eachSize, eachSize), Image.ANTIALIAS)
    Real.paste(d_img, (x * eachSize, y * eachSize))
    x += 1
    if x == eachLine*2:
        x = 0
        y += 1
plt.imshow(Real)

plt.show()