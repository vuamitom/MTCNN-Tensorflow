#coding:utf-8
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from train_models.MTCNN_config import config

# sys.path.append("../prepare_data")
# print(sys.path)
from prepare_data.read_tfrecord_v2 import read_multi_tfrecords,read_single_tfrecord
import math
import random
import cv2

no_landmarks = 68


def train_model(base_lr, loss, data_num, quantize=True, optimizer_type='momentum'):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    #LR_EPOCH [8,14]
    #boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    #lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    #control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    # quantize graph 
    if quantize:
        print('quantize ', quantize)
        tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=20000)
    print ('optimizer_type', optimizer_type)
    if optimizer_type == 'adam':
        # optimizer = tf.train.AdamOptimizer(lr_op, 0.9)
        print('adam')
        optimizer = tf.train.AdamOptimizer(lr_op, 0.9, 0.999)
    else:
        print('momentum')
        optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    # check_op = tf.add_check_numerics_ops()
    return train_op, lr_op

'''
certain samples mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    num_images = image_batch.shape[0]
    random_number = npr.choice([0,1],num_images,replace=True)
    #the index of image needed to flip
    indexes = np.where(random_number>0)[0]
    fliplandmarkindexes = np.where(label_batch[indexes]==-2)[0]
    
    #random flip    
    for i in indexes:
        cv2.flip(image_batch[i],1,image_batch[i])
    #pay attention: flip landmark    
    for i in fliplandmarkindexes:
        landmark_ = landmark_batch[i].reshape((-1,2))
        landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
        landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
        landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
        landmark_batch[i] = landmark_.ravel()
    return image_batch,landmark_batch
'''
# all mini-batch mirror

def flip_indices():
    """
    refer to https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
    """
    r = []
    if no_landmarks == 68:
    
        # chin
        for i in range(0, 8):
            r.append([i, 16 - i])
        # forhead
        for i in range(17, 22):
            r.append([i, 43 - i])
        # upper eyes
        for i in range(36, 40):
            r.append([i, 81 - i])

        # lower eyes
        r.append([41, 46])
        r.append([40, 47])

        # nose
        r.append([31, 35])
        r.append([32, 34])

        # lips
        r.append([48, 54])
        r.append([49, 53])
        r.append([50, 52])
        r.append([60, 64])
        r.append([61, 63])
        r.append([59, 55])
        r.append([67, 65])
        r.append([58, 56])
    else:
        r.append([0, 1])
        r.append([3, 4])
    return r

landmark_indices_to_flip = flip_indices()

def random_flip_images(image_batch,label_batch,landmark_batch):
    #mirror
    if random.choice([0,1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip landmark    
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            # landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            # landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth
            for pair in landmark_indices_to_flip:
                reverse_pair = [pair[1], pair[0]]
                landmark_[pair] = landmark_[reverse_pair]        
            landmark_batch[i] = landmark_.ravel()
        # print('flipindices', flipindexes.shape, flipindexes)
        # print('fliplandmarkindexes', fliplandmarkindexes.shape, fliplandmarkindexes)
        # print('flipposindexes', flipposindexes.shape, flipposindexes)
        # exit(0)
        # test_img = image_batch[flipindexes[0]] * 128 + 127.5        
        # cv2.imshow('test', test_img)
        # cv2.waitKey(0)
    return image_batch,landmark_batch

def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)

    return inputs

def count_nan(v):
    # print('v = ', v)
    return np.sum(np.where(np.isnan(v), np.ones_like(v), np.zeros_like(v)))
    # return tf.reduce_sum(tf.where(tf.is_nan(v), tf.ones_like(v), tf.zeros_like(v)))

def train(net_factory, prefix, end_epoch, base_dir, log_dir,
          display=200, base_lr=0.01, quantize=True, ckpt=None, optimizer='momentum'):
    """
    train PNet/RNet/ONet
    :param net_factory:
    :param prefix: model path
    :param end_epoch:
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """
    print('start training: ....')
    net = prefix.split('/')[-1]
    #label file
    label_file = os.path.join(base_dir,'train_%s_landmark.txt' % net)
    #label_file = os.path.join(base_dir,'landmark_12_few.txt')
    print(label_file)
    f = open(label_file, 'r')
    # get number of training examples
    num = len(f.readlines())
    print("Total size of the dataset is: ", num)
    print(prefix)

    #PNet use this method to get data
    if net == 'PNet':
        #dataset_dir = os.path.join(base_dir,'train_%s_ALL.tfrecord_shuffle' % net)
        dataset_dir = os.path.join(base_dir,'train_%s_landmark.tfrecord_shuffle' % net)
        print('dataset dir is:',dataset_dir , 'batch_size = ', config.BATCH_SIZE)
        image_batch, label_batch, bbox_batch, landmark_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net, no_landmarks)
        
    #RNet use 3 tfrecords to get data    
    else:
        pos_dir = os.path.join(base_dir,'pos_landmark.tfrecord_shuffle')
        part_dir = os.path.join(base_dir,'part_landmark.tfrecord_shuffle')
        neg_dir = os.path.join(base_dir,'neg_landmark.tfrecord_shuffle')
        #landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
        landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
        dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]
        pos_radio = 1.0/6;part_radio = 1.0/6;landmark_radio=1.0/6;neg_radio=3.0/6
        pos_batch_size = int(np.ceil(config.BATCH_SIZE * pos_radio))
        assert pos_batch_size != 0,"Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE * part_radio))
        assert part_batch_size != 0,"Batch Size Error "
        neg_batch_size = int(np.ceil(config.BATCH_SIZE * neg_radio))
        assert neg_batch_size != 0,"Batch Size Error "
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE * landmark_radio))
        assert landmark_batch_size != 0,"Batch Size Error "
        batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
        #print('batch_size is:', batch_sizes)
        image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net, no_landmarks)        
        
    #landmark_dir    
    if net == 'PNet':
        image_size = 12
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    else:
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 1;
        image_size = 48
    
    #define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE, no_landmarks *2],name='landmark_target')
    #get loss and accuracy
    input_image = image_color_distort(input_image)
    cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op, landmark_pred = net_factory(input_image, label, bbox_target,landmark_target,training=True)
    #train,update learning rate(3 loss)
    # count_nan_op = count_nan(landmark_pred)
    total_loss_op  = radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_landmark_loss* landmark_loss_op + L2_loss_op
    train_op, lr_op = train_model(base_lr,
                                  total_loss_op,
                                  num, quantize, optimizer)
    

    # init
    sess = tf.Session()

    #save model
    saver = tf.train.Saver(max_to_keep=10)
    step = 0
    if ckpt is not None:
        saver.restore(sess, ckpt)
        # get last global step 
        step = int(os.path.basename(ckpt).split('-')[1])
        print('restored from last step = ', step)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("total_loss",total_loss_op)#cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()
    logs_dir = os.path.join(log_dir, net)
    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer,projector_config)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # i = 0
    
    #total steps
    step_per_epoch = int(num / config.BATCH_SIZE + 1)
    print ('step_per_epoch = ', step_per_epoch)
    MAX_STEP = step_per_epoch * end_epoch
    epoch = 0
    sess.graph.finalize()
    current_total_loss = 100000
    try:
        for i in range(MAX_STEP):
            # i = i + 1
            # j = i
            step = step + 1
            if coord.should_stop():
                break
            # print ('train step = ', step, image_batch.shape, bbox_batch.shape, landmark_batch.shape)
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            #random flip
            # print('after batch array')
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)

            # print('->>>>> 1')
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})

            if (step+1) % display == 0:
                #acc = accuracy(cls_pred, labels_batch)
                # print('->>>>> 2') 
                cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc, landmark_pred_val = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op, landmark_pred],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})
                if math.isnan(landmark_loss):
                    print('break, landmark loss is nan', landmark_loss)
                    print('landmark pred val ', landmark_pred_val)
                    # nan_count = sess.run([count_nan_op], feed_dict={landmark_pred: landmark_pred_val})
                    print('no of nan in landmark_pred_val', count_nan(landmark_pred_val))
                    print('no of nan in landmark_target', count_nan(landmark_batch_array))
                    # print('other metrics ', square_error_val, k_index_val, valid_inds_val)
                    break
                    
                total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_landmark_loss*landmark_loss + L2_loss
                # landmark loss: %4f,
                print("%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                datetime.now(), step+1,MAX_STEP, acc, cls_loss, bbox_loss,landmark_loss, L2_loss,total_loss, lr))
                if total_loss < current_total_loss:
                    current_total_loss = total_loss
                    path_prefix = saver.save(sess, prefix, global_step=step)
                    print ('Total loss improved, save model ', path_prefix)
            # save every end of epochs
            # if i > 0 and i % step_per_epoch == 0:                
            #     path_prefix = saver.save(sess, prefix, global_step=step)
            #     print('Save end of epoch, path prefix is :', path_prefix)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
