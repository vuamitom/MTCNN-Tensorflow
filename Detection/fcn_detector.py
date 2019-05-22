import numpy as np
import tensorflow as tf
import sys
sys.path.append("../")
from train_models.MTCNN_config import config


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):
        #create a graph
        self.is_quantized = False 

        graph = tf.Graph()
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            #self.cls_prob batch*2
            #self.bbox_pred batch*4
            #construct model here
            #self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #contains landmark
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            
            #allow 
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)
    def predict(self, databatch):
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        return cls_prob, bbox_pred

class TFLiteFcnDetector(object):
    def __init__(self, model_path, input_img_size):
        #create a graph
        self.is_quantized = True
        self.interpreters = {}
        sizes = None
        if input_img_size == 240:
            sizes = [14, 18, 23, 29, 37, 47, 59, 75, 95, 120]
        elif input_img_size == 120:
            sizes = [15, 18, 23, 30, 37, 47, 60]
        for s in sizes:
            ip = tf.lite.Interpreter(model_path=(model_path + '_' + str(s) + '.tflite'))
            ip.allocate_tensors()
            self.interpreters[s] = ip
            
    def predict(self, databatch):
        height, width, _ = databatch.shape
        # print(height, width)
        net = self.interpreters[height]
        input_details = net.get_input_details()
        output_details = net.get_output_details()
        # print('databatch ', databatch)
        ds = databatch.reshape(1, *databatch.shape)
        net.set_tensor(input_details[0]['index'], ds.astype(np.float32))
        net.invoke()

        output_dict = {
            'cls_prob': net.get_tensor(output_details[0]['index']),
            'bbox_pred': net.get_tensor(output_details[1]['index'])           
        }
        return output_dict['cls_prob'], output_dict['bbox_pred']
