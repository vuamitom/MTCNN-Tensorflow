import sys
sys.path.append('../..')
import tensorflow as tf
import os 
from train_models.mtcnn_model import P_Net, R_Net, O_Net

def export_model(path, data_size, net_factory):
    if data_size is None:
        return
    print ('exporting model ', path, 'data_size=', data_size)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            img = tf.placeholder(tf.float32, shape=[1, data_size, data_size, 3], name='input_image')
            cls_prob, bbox_pred, landmark_pred = net_factory(img, training=False)
            saver = tf.train.Saver()
            saver.restore(sess, path)
            converter = tf.lite.TFLiteConverter.from_session(sess, [img], [cls_prob, bbox_pred, landmark_pred])
            tflite_model = converter.convert()
            with open(os.path.join(os.path.dirname(path), os.path.basename(path).split('-')[0] + '.tflite'), 'wb') as f:
                f.write(tflite_model)


prefix = ['PNet_landmark/PNet', 'RNet_landmark/RNet', 'ONet_landmark/ONet']
epoch = [18, 14, 16]
data_sizes = [None, 24, 48]
net_factories = [P_Net, R_Net, O_Net]
model_paths = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

print(model_paths)
for idx in range(0, len(model_paths)):
    export_model(model_paths[idx], data_sizes[idx], net_factories[idx])