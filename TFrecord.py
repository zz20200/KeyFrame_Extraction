# -*- coding:utf-8 -*-
'''
##todo:###in this program, we start to make train data with tf-records,
##todo:###
作者：张治强
程序：制作TFRecords格式文件
时间：2018年9月22日17:01:02
'''
import os
import tensorflow as tf
import cv2
import sys

##todo:make data with tf-records，制作数据集时乱序制作___________优先使用此子函数
def creat_record(filename,path_file,size):
    writer = tf.python_io.TFRecordWriter(filename)
    img_name  = os.listdir(path_file)
    Num = len(img_name)
    for j in range(Num):
        try:
            print("j",j)
            img = cv2.imread(path_file + "//"+img_name[j],cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,size)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
        except:
            continue
    writer.close()

def read_and_decode(filename,length):
    # filename_queue = tf.train.string_input_producer([filename])
    filename_queue = filename
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features ={
        'img_raw':tf.FixedLenFeature([],tf.string),
    })
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape(img,[length])
    img = tf.cast(img, tf.float32) * (1./128) - 1#处理
    return img


# path_cwd1 = os.path.abspath(sys.argv[0])
# path_cwd,_ = os.path.split(path_cwd1)


def make():
    ####制作训练集1
    train_filename = "train_zz.tfrecords"
    path_file1 = r"C:\shen\random_extracted_img"
    creat_record(train_filename,path_file1,(252,252))

# make()
