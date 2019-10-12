import tensorflow as tf
import numpy as np
import read
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def conv_batch_norm(x, n_out, train):
    beta = tf.get_variable("beta", [n_out], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
    gamma = tf.get_variable("gamma", [n_out], initializer=tf.constant_initializer(value=1.0, dtype=tf.float32))
    
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(train, mean_var_with_update, lambda:(ema_mean, ema_var))
    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)

    #mean_hist = tf.summary.histogram("meanHistogram", mean)
    #var_hist = tf.summary.histogram("varHistogram", var)
    return normed

def layer_batch_norm(x, n_out, is_train):
    beta = tf.get_variable("beta", [n_out], initializer=tf.zeros_initializer())# 
    gamma = tf.get_variable("gamma", [n_out], initializer=tf.ones_initializer())# 

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')#
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(is_train, mean_var_with_update, lambda:(ema_mean, ema_var))

    x_r = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(x_r, mean, var, beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])

def conv2d(input, weight_shape):
    size = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weights_init = tf.random_normal_initializer(stddev=np.sqrt(2. / size))
    biases_init = tf.zeros_initializer()
    weights = tf.get_variable(name="weights", shape=weight_shape, initializer=weights_init)#
    biases = tf.get_variable(name="biases", shape=weight_shape[3], initializer=biases_init)# 

    conv_out = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv_add = tf.nn.bias_add(conv_out, biases)
    conv_batch = conv_batch_norm(conv_add, weight_shape[3], tf.constant(True, dtype=tf.bool))
    output = tf.nn.relu(conv_batch)

    return output

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def layer(x, weights_shape):
    init = tf.random_normal_initializer(stddev=np.sqrt(2. / weights_shape[0]))
    weights = tf.get_variable(name="weights", shape=weights_shape, initializer=init)
    biases = tf.get_variable(name="biases", shape=weights_shape[1], initializer=init)
    mat_add = tf.matmul(x, weights) + biases
    mat_batch = layer_batch_norm(mat_add, weights_shape[1], tf.constant(True, dtype=tf.bool))
    output = tf.nn.relu(mat_batch)
    
    return output

# [filter size, filter height, filter weight, filter depth]
conv1_size = [5, 5, 1, 32]
conv2_size = [5, 5, 32, 64]
# The new size(7 * 7) before image(28 * 28) used the pooling(2 times) method
hide3_size = [80 * 80 * 64, 1024]
output_size = 10

def predict(x, keep_drop):
    x = tf.reshape(x, shape=[-1, 320, 320, 1])
    with tf.variable_scope("conv1_scope"):
        conv1_out = conv2d(x, conv1_size)
        pool1_out = max_pool(conv1_out)

    with tf.variable_scope("conv2_scope"):
        conv2_out = conv2d(pool1_out, conv2_size)
        pool2_out = max_pool(conv2_out)

    with tf.variable_scope("hide3_scope"):
        pool2_flat = tf.reshape(pool2_out, [-1, hide3_size[0]])
        hide3_out = layer(pool2_flat, hide3_size)
        #hide3_drop = tf.nn.dropout(hide3_out,keep_drop)

    with tf.variable_scope("out_scope"):
        output = layer(hide3_out, [hide3_size[1], output_size])

    return output

        
def loss(y, t):
    cross = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t)
    result = tf.reduce_mean(cross)
    #loss_his = tf.summary.scalar("loss", result)

    return result

def train(loss, index):
    return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss, global_step=index)

def accuracy(output, t):
    comparison = tf.equal(tf.argmax(output, 1), tf.argmax(t, 1))
    y = tf.reduce_mean(tf.cast(comparison, tf.float32))

    tf.summary.scalar("accuracy error", (1.0 - y))

    return y

batch_size = 50
train_times = 10
train_step = 1
train_number =10000

if __name__ == '__main__':
    # init
    train_data,test_data = read.getdata()
    mnist = read_data_sets("MNIST/", one_hot=True)
    input_x = tf.placeholder(tf.float32, shape=[None, 102400], name="input_x")
    input_y = tf.placeholder(tf.float32, shape=[None, 10], name="input_y")
    print(input_x)
    print(input_y)
    
    # predict
    predict_op = predict(input_x, 0.5)

    # loss
    loss_op = loss(predict_op, input_y)

    # train
    index = tf.Variable(0, name="train_time")
    train_op = train(loss_op, index)

    # accuracy
    accuracy_op = accuracy(predict_op, input_y)

    # graph
    summary_op = tf.summary.merge_all()
    session = tf.Session()
    summary_writer = tf.summary.FileWriter("log/", graph=session.graph)

    init_value = tf.global_variables_initializer()
    session.run(init_value)
    print(session.run(init_value))
    saver = tf.train.Saver()

    for time in range(train_times):
        avg_loss = 0.
        total_batch = int(train_number / batch_size)
        #print("mnist.train.num_examples",mnist.train.num_examples) #55000 50000 for train 5000 for test
        for i in range(total_batch):
            minibatch_x=np.array([[]])            
            for j in range(10):
                for k in range(5):
                    if(j==0 and k==0):
                        c=np.array([train_data[j][i*5+k]])
                        minibatch_x=c.reshape(1,102400)
                        temp=np.zeros([10])
                        temp[j]=1
                        minibatch_y=np.array([temp])
                    else:
                        c=np.array([train_data[j][i*5+k]])
                        minibatch_x=np.concatenate((minibatch_x,c.reshape(1,102400)), axis=0)
                        temp=np.zeros([10])
                        temp[j]=1
                        minibatch_y=np.concatenate((minibatch_y, np.array([temp])), axis=0)
            print(type(minibatch_x),type(minibatch_y))
            print(minibatch_x.shape,minibatch_y.shape)
            minibatch_x2, minibatch_y2 = mnist.train.next_batch(batch_size)
            print(type(minibatch_x2),type(minibatch_y2))
            print(minibatch_x2.shape,minibatch_y2.shape)
            #print("x,y",len(minibatch_x[0]),len(minibatch_y[0])) 28*28*128 10*128
            #print("x,y",minibatch_x[0][100:110],len(minibatch_y[0])) x is normalized from 0-1
            session.run(train_op, feed_dict={input_x: minibatch_x, input_y: minibatch_y})
            avg_loss += session.run(loss_op, feed_dict={input_x: minibatch_x, input_y: minibatch_y}) / total_batch

        if (time + 1) % train_step == 0:
            for i in range(10):
                if(i==0):
                    temp=np.zeros([400,10])
                    temp[:,i]=1
                    test_data_label=temp
                else:
                    temp=np.zeros([400,10])
                    temp[:,i]=1
                    test_data_label=np.concatenate((test_data_label,temp),axis=0)

            #accuracy = session.run(accuracy_op, feed_dict={input_x: mnist.validation.images, input_y: mnist.validation.labels})
            accuracy = session.run(accuracy_op, feed_dict={input_x:test_data, input_y: test_data_label})
            #summary_str = session.run(summary_op, feed_dict={input_x: mnist.validation.images, input_y: mnist.validation.labels})
            summary_str = session.run(summary_op, feed_dict={input_x: test_data, input_y: test_data_label})
            #print("mnist.validation.images",len(mnist.validation.images[0])) 28*28*5000
            #print("mnist.validation.labels",len(mnist.validation.labels[0])) 10*5000
            summary_writer.add_summary(summary_str, session.run(index))
            print("train times:", (time + 1),
                        " avg_loss:", avg_loss,
                        " accuracy:", accuracy)

    y = session.run(predict_op, feed_dict={input_x:test_data[0:100]})
    print("predict : " + str(np.argmax(y, axis=1)))
    print("really: " + str(np.argmax(test_data_label[0:100], axis=1)))
    #plt.imshow((mnist.validation.images[0].reshape(28, 28)))
    #plt.show()

    session.close()
