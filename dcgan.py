import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
def generator(x,is_training,channels):
    depths=[512,256,128,64,channels]
    original_img_size=3
    
    
    inputs=tf.convert_to_tensor(x)
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('reshape'):
            #batch*10----batch*512*3*3
            dense=tf.layers.dense(inputs,depths[0]*original_img_size*original_img_size)
            reshape=tf.reshape(dense,[-1,original_img_size,original_img_size,depths[0]])
            bn = tf.nn.relu(tf.contrib.layers.batch_norm(reshape,\
                                is_training=is_training), name='bn')           
        with tf.variable_scope('transpose1'):
            #batch*512*3*3----batch*256*6*6
            transpose=tf.contrib.layers.conv2d_transpose(inputs=bn,\
                            num_outputs=depths[1],kernel_size=4,stride=(2, 2),padding='SAME')
            bn = tf.nn.relu(tf.contrib.layers.batch_norm(transpose,\
                                is_training=is_training), name='bn') 
        with tf.variable_scope('transpose2'):
            #batch*256*6*6----batch*128*12*12
            transpose=tf.contrib.layers.conv2d_transpose(inputs=bn,\
                            num_outputs=depths[2],kernel_size=4,stride=(2, 2),padding='SAME')
            bn = tf.nn.relu(tf.contrib.layers.batch_norm(transpose,\
                                is_training=is_training), name='bn') 
        with tf.variable_scope('transpose3'):
            #batch*128*12*12----batch*64*24*24
            transpose=tf.contrib.layers.conv2d_transpose(inputs=bn,\
                            num_outputs=depths[3],kernel_size=4,stride=(2, 2),padding='SAME')
            bn = tf.nn.relu(tf.contrib.layers.batch_norm(transpose,\
                                is_training=is_training), name='bn') 
        with tf.variable_scope('transpose4'):
            #batch*64*24*24----batch*3*28*28 by valid padding
        
            transpose=tf.contrib.layers.conv2d_transpose(inputs=bn,\
                            num_outputs=depths[4],kernel_size=5,stride=(1, 1),padding='VALID')
            bn = bn = tf.nn.relu(tf.contrib.layers.batch_norm(transpose,\
                                is_training=is_training), name='bn') 
        with tf.variable_scope('tanh'):
            outputs = tf.tanh(x=bn, name='outputs')
    return outputs

def discriminator(x,is_training,channels,num_classes):
    inputs = tf.convert_to_tensor(x)
    depths=[channels,64,128,256,512]
    with  tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('conv1'):
            conv=tf.contrib.layers.conv2d(inputs=inputs,\
                            num_outputs=depths[1],kernel_size=5,stride=(1, 1),padding='VALID')
            bn=tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv,\
                                is_training=is_training), name='bn')
        with tf.variable_scope('conv2'):
            conv=tf.contrib.layers.conv2d(inputs=inputs,\
                            num_outputs=depths[2],kernel_size=3,stride=(2, 2),padding='SAME')
            bn=tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv,\
                                is_training=is_training), name='bn')
        with tf.variable_scope('conv3'):
            conv=tf.contrib.layers.conv2d(inputs=inputs,\
                            num_outputs=depths[3],kernel_size=3,stride=(2, 2),padding='SAME')
            bn=tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv,\
                                is_training=is_training), name='bn')

        with tf.variable_scope('conv4'):
            conv=tf.contrib.layers.conv2d(inputs=inputs,\
                            num_outputs=depths[4],kernel_size=3,stride=(2, 2),padding='SAME')
            bn=tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv,\
                                is_training=is_training), name='bn')
        with tf.variable_scope('classify'):
                batch_size = bn.get_shape()[0].value
                reshape = tf.reshape(bn, [batch_size, -1])
                outputs = tf.layers.dense(reshape, num_classes, name='outputs')
        return outputs
def save_img(tmp_batch,i):
    #x is an array of an image
    for j in range(0,len(tmp_batch)):
        img=tf.keras.preprocessing.image.array_to_img(tmp_batch[j])
        img.save('./generated/'+str(i)+'_'+str(j)+'.jpg','JPEG')
       
batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS=64,28,28,1
noise_dim=20
channels=1
epochs=20000
num_classes=1
mnist = input_data.read_data_sets('MNIST_data')
with tf.device('/gpu:0'):
    real_input = tf.placeholder(tf.float32, [batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    noise_input = tf.placeholder(tf.float32, [batch_size, noise_dim])
	#One step G, one step D for real images, and one step D for fake images from G.
    G_logits = generator(x=noise_input,is_training=True,channels=channels)
    
    d_fake_pred = discriminator(x=G_logits,\
                is_training=True,channels=channels,num_classes=num_classes)

    d_real_pred = discriminator(x=real_input,\
                is_training=True,channels=channels,num_classes=num_classes)
    
    
    with tf.name_scope('Loss'):    
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits =d_real_pred, labels = tf.ones_like(d_real_pred)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_pred, labels = tf.zeros_like(d_fake_pred)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_pred, labels = tf.ones_like(d_fake_pred)))
        d_loss = d_loss_real + d_loss_fake
        tf.summary.scalar('d_loss',d_loss)
        tf.summary.scalar('g_loss',g_loss)
    with tf.name_scope('G_optimizer'):
        g_optimizer=tf.train.AdamOptimizer(0.0002,0.5).minimize(loss=g_loss)
    with tf.name_scope('D_optimizer'):
        d_optimizer=tf.train.AdamOptimizer(0.0002,0.5).minimize(loss=d_loss)
    
    
    merge=tf.summary.merge_all()
    sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess.run(tf.global_variables_initializer() )
    writer=tf.summary.FileWriter('./logs',sess.graph)
    saver=tf.train.Saver()
    
    ckpt=tf.train.get_checkpoint_state('./DCGAN_model')
    if ckpt!=None:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('Model restored')
    else:
        print('Created a new model')
    for i in range(epochs):
        noise = np.random.uniform(0.0, 1.0, [batch_size, noise_dim]).astype(np.float32)
        batch = [np.reshape(b, [28,28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]  
        batch=np.array(batch)
        
        _,loss1=sess.run([d_optimizer,d_loss],feed_dict={real_input:batch,noise_input:noise})
        _,loss2=sess.run([g_optimizer,g_loss],feed_dict={noise_input:noise})
#        loss1=sess.run(d_loss,{real_input:batch,noise_input:noise})
#        loss2=sess.run(g_loss,{noise_input:noise})
        print('epoch=',i,'d_loss=',loss1,'g_loss=',loss2)
        if i%1000==0 and i>0:
            merged=sess.run(merge,{real_input:batch,noise_input:noise})
            writer.add_summary(merged,i)
            tmp_batch=sess.run(G_logits,{noise_input:noise})
            plt.imshow(tmp_batch[0].reshape([28,28]))
            save_img(tmp_batch,i)
            
                
    
    saver.save(sess,'./DCGAN_model/model')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    