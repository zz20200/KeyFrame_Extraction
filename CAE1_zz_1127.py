import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.contrib.layers.python.layers import batch_norm
import TFrecord as TR

n_input = 252 * 252 #输入的维度
batch_size = 10##每批的数目
im_size= 252

def downSampleLayer(x_input,shape,isTraining,strides,layer_name,padding_type = 'SAME',activation_func=None):
    w = tf.get_variable(name= layer_name+"w",shape=shape,dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
    print (w)
    b = tf.get_variable(name= layer_name+"b",shape=[ shape[3] ],dtype=tf.float32,initializer=tf.constant_initializer(0.01) )
    result_conv = tf.nn.conv2d(input=x_input,filter=w,strides=strides,padding=padding_type,name=layer_name+"conv")
    result_add = tf.nn.bias_add(value=result_conv,bias=b,name=layer_name+'add')
    result_BN = batch_norm(inputs=result_add,decay=0.9,is_training=isTraining)

    if activation_func is None:
        outputs = result_BN
    else:
        outputs = activation_func(result_BN,name=layer_name+"act")
    return outputs

def upSampleLayer(x_input,shape,output_shape,strides,isTraining,layer_name,padding_type = 'SAME',activation_func=None):
    w = tf.get_variable(name=layer_name + "w", shape=shape, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable(name=layer_name + "b", shape=[shape[2]], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.01))
    result_dconv = tf.nn.conv2d_transpose(value=x_input, filter=w, output_shape=output_shape,
                                          strides=strides, padding=padding_type, name=layer_name+"dconv")
    result_add = tf.nn.bias_add(value=result_dconv, bias=b, name=layer_name + 'add')
    result_BN = batch_norm(inputs=result_add, decay=0.9, is_training=isTraining)

    if activation_func is None:
        outputs = result_BN
    else:
        outputs = activation_func(result_BN,name=layer_name+"act")
    return outputs

x = tf.placeholder(dtype=tf.float32,shape=[None,n_input],name="input_imgs")
x_input = tf.reshape(tensor=x,shape=[-1,im_size,im_size,1],name="input")

result_layer1 = downSampleLayer(x_input=x_input,      shape=[3,3,1,64],     isTraining=True,layer_name="layer1",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##1*252*252--64*250*250
result_layer2 = downSampleLayer(x_input=result_layer1,shape=[3,3,64,64],    isTraining=True,layer_name="layer2",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##64*250*250--64*248*248
result_layer3 = downSampleLayer(x_input=result_layer2,shape=[2,2,64,128],   isTraining=True,layer_name="layer3",
                                strides=[1,2,2,1],padding_type='VALID',activation_func=tf.nn.elu)##64*248*248-128*124*124

result_layer4 = downSampleLayer(x_input=result_layer3,shape=[3,3,128,128],  isTraining=True,layer_name="layer4",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##128*124*124--128*122*122
result_layer5 = downSampleLayer(x_input=result_layer4,shape=[3,3,128,128],  isTraining=True,layer_name="layer5",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##128*122*122--128*120*120
result_layer6 = downSampleLayer(x_input=result_layer5,shape=[2,2,128,256],  isTraining=True,layer_name="layer6",
                                strides=[1,2,2,1],padding_type='VALID',activation_func=tf.nn.elu)##128*120*120-256*60*60

result_layer7 = downSampleLayer(x_input=result_layer6,shape=[3,3,256,256],  isTraining=True,layer_name="layer7",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##256*60*60-256*58*58
result_layer8 = downSampleLayer(x_input=result_layer7,shape=[3,3,256,256],  isTraining=True,layer_name="layer8",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##256*58*58-256*56*56


result_layerB1 = downSampleLayer(x_input=result_layer8,shape=[1,1,256,128],  isTraining=True,layer_name="layerB1",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##256*56*56-256*56*56

result_layerB2 = downSampleLayer(x_input=result_layerB1,shape=[1,1,128,64],  isTraining=True,layer_name="layerB2",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##128*56*56-64*56*56

result_layerB3 = downSampleLayer(x_input=result_layerB2,shape=[1,1,64,1],  isTraining=True,layer_name="layerB3",
                                strides=[1,1,1,1],padding_type='VALID',activation_func=tf.nn.elu)##64*56*56-1*56*56

#############TODO: ---------------------------------------------------------------------------------------------------------------------------------------

result_layerA1 = upSampleLayer(x_input=result_layerB3,shape=[1,1,64,1],output_shape=[batch_size,56,56,64],
                              strides=[1,1,1,1],isTraining=True,layer_name='layerA1',padding_type='VALID',activation_func=tf.nn.elu)##512*26*26-512*28*28

result_layerA2 = upSampleLayer(x_input=result_layerA1,shape=[1,1,128,64],output_shape=[batch_size,56,56,128],
                              strides=[1,1,1,1],isTraining=True,layer_name='layerA2',padding_type='VALID',activation_func=tf.nn.elu)##512*26*26-512*28*28

result_layerA3 = upSampleLayer(x_input=result_layerA2,shape=[1,1,256,128],output_shape=[batch_size,56,56,256],
                              strides=[1,1,1,1],isTraining=True,layer_name='layerA3',padding_type='VALID',activation_func=tf.nn.elu)##512*26*26-512*28*28



result_layer13 = upSampleLayer(x_input=result_layerA3,shape=[3,3,256,256],output_shape=[batch_size,58,58,256],
                              strides=[1,1,1,1],isTraining=True,layer_name='layer13',padding_type='VALID',activation_func=tf.nn.elu)##256*56*56-256*58*58
result_layer14 = upSampleLayer(x_input=result_layer13,shape=[3,3,256,256],output_shape=[batch_size,60,60,256],
                              strides=[1,1,1,1],isTraining=True,layer_name='layer14',padding_type='VALID',activation_func=tf.nn.elu)##256*58*58-128*60*60

result_layer15 = upSampleLayer(x_input=result_layer14,shape=[2,2,128,256],output_shape=[batch_size,120,120,128],
                              strides=[1,2,2,1],isTraining=True,layer_name='layer15',padding_type='VALID',activation_func=tf.nn.elu)##128*60*60-128*120*120
result_layer16 = upSampleLayer(x_input=result_layer15,shape=[3,3,128,128],output_shape=[batch_size,122,122,128],
                              strides=[1,1,1,1],isTraining=True,layer_name='layer16',padding_type='VALID',activation_func=tf.nn.elu)##128*120*120-128*122*122
result_layer17 = upSampleLayer(x_input=result_layer16,shape=[3,3,128,128],output_shape=[batch_size,124,124,128],
                              strides=[1,1,1,1],isTraining=True,layer_name='layer17',padding_type='VALID',activation_func=tf.nn.elu)##128*122*122-64*124*124

result_layer18 = upSampleLayer(x_input=result_layer17,shape=[2,2,64,128],output_shape=[batch_size,248,248,64],
                              strides=[1,2,2,1],isTraining=True,layer_name='layer18',padding_type='VALID',activation_func=tf.nn.elu)##64*124*124-64*248*248
result_layer19 = upSampleLayer(x_input=result_layer18,shape=[3,3,64,64],output_shape=[batch_size,250,250,64],
                              strides=[1,1,1,1],isTraining=True,layer_name='layer19',padding_type='VALID',activation_func=tf.nn.elu)##64*250*250-64*250*250
result_layer20 = upSampleLayer(x_input=result_layer19,shape=[3,3,1,64],output_shape=[batch_size,252,252,1],
                              strides=[1,1,1,1],isTraining=True,layer_name='layer20',padding_type='VALID',activation_func=tf.nn.elu)##64*250*250-1*252*252

loss = tf.reduce_mean( tf.reduce_sum( tf.abs(result_layer20 - x_input),reduction_indices=[1,2] ) )

LRRate = 0.001
train_step = tf.train.AdamOptimizer(LRRate).minimize(loss)

def train():
    file_name = 'trainZZ.tfrecords'
    filename_queue = tf.train.string_input_producer([file_name],num_epochs=None)
    trainData = TR.read_and_decode(filename_queue,252*252)
    train_batch = tf.train.shuffle_batch([trainData],batch_size=batch_size,capacity=50,min_after_dequeue=30)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        for i in range(3000):
            train_example = sess.run([train_batch])
            # print(train_example[0].shape)
            train_step.run(session=sess,feed_dict={x:train_example[0]})
            if i % 10 == 0:
                loss1 = sess.run(loss,feed_dict={x:train_example[0]})
                print("sss",train_example[0].shape)
                im = sess.run(result_layer20,feed_dict={x:train_example[0]})
                im2= sess.run(result_layerB3,feed_dict={x:train_example[0]})
                print("sss", im2.shape,im2[0][:,:,0].shape)
                im = im[0][:, :, 0]
                im = (im + 1)*128

                im2 = im2[0][:,:,0]
                im2 = (im2 + 1) *128
                cv2.imwrite("imresutl.jpg",im)
                cv2.imwrite("2.jpg",im2)
                print("训练次数",i,"loss",loss1)
                if i % 40 == 0:
                    saver.save(sess, save_path)
    except tf.errors.OutOfRangeError:
        print("done reading train data")

    finally:
        coord.request_stop()
    coord.request_stop()
    coord.join(threads)



def try_extractfeature(im):
    im = (im / 128) - 1
    im = np.reshape(im,(1,63504))

    pre = sess.run(result_layerB3, feed_dict={x: im})
    pre = np.reshape(pre, (1, 56, 56))
    pre1 = (pre[0, :, :] + 1) * 128
    pre1= np.array(pre1).reshape((56,56))
    return pre1


sess = tf.Session()
saver = tf.train.Saver()
save_path = r"I:\KeyFrame_11_29\model_zz\model.ckpt"
saver.restore(sess, save_path)
# train()


# os.chdir(r"C:\shen\random_extracted_img")
# name = "96.bmp"
# im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
# try_extractfeature(im)