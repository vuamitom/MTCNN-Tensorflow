import sys
sys.path.append('../..')
import tensorflow as tf
import os 
from train_models.mtcnn_model import P_Net, R_Net, O_Net

def export_model(path, data_size, net_factory, output_name, quantized=False):
    # if data_size is None:
    #     return
    print ('exporting model ', path, 'data_size=', data_size)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            # inputs, outputs = None, None
            # if data_size is None:
            #     img = tf.placeholder(tf.float32, name='input_image')
            #     width = tf.placeholder(tf.float32, name='image_width')
            #     height = tf.placeholder(tf.float32, name='image_height')
            #     image_reshape = tf.reshape(img, [1, height, width, 3])
            #     inputs = [img, width, height]
            #     cls_prob, bbox_pred, landmark_pred = net_factory(image_reshape, training=False)
            #     outputs = [cls_prob, bbox_pred, landmark_pred]
            # else:
            img = tf.placeholder(tf.float32, shape=[1, data_size, data_size, 3], name='input_image')
            cls_prob, bbox_pred, landmark_pred = net_factory(img, training=False)
            inputs = [img]
            outputs = [cls_prob, bbox_pred, landmark_pred]

            saver = tf.train.Saver()
            saver.restore(sess, path)
            converter = tf.lite.TFLiteConverter.from_session(sess, inputs, outputs)
            if quantized:
                converter.inference_type = tf.uint8
                converter.allow_custom_ops = True
                converter.quantized_input_stats = {}
                converter.quantized_input_stats[inputs[0]] = (128, 128) # (mean, std)
            tflite_model = converter.convert()
            op = os.path.join(os.path.dirname(path), output_name + '.tflite')
            with open(op, 'wb') as f:
                f.write(tflite_model)
            print('write to ', op)

prefix = ['RNet_landmark/RNet', 'ONet_landmark/ONet']
epoch = [14, 16]
data_sizes = [24, 48]
net_factories = [R_Net, O_Net]
model_paths = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
outputs = ['RNet', 'ONet']
# get necessary pnet size 
def gen_pnet_sizes():
    init_bitmap_size = 240
    pnet_size = 12
    scale_factor = 0.79
    min_face_size = 24
    current_scale = pnet_size / float(min_face_size)
    all_sizes = []
    bitmap_size = round(init_bitmap_size * current_scale)
    while bitmap_size > pnet_size:
        print('bm si = ', bitmap_size)
        all_sizes.append(bitmap_size)
        current_scale *= scale_factor
        bitmap_size = round(init_bitmap_size * current_scale)        
        
    return all_sizes

for s in gen_pnet_sizes():
    data_sizes.append(s)
    net_factories.append(P_Net)
    model_paths.append('PNet_landmark/PNet-18')
    outputs.append('PNet_%s' % (s,))

print(model_paths)
for idx in range(0, len(model_paths)):
    export_model(model_paths[idx], data_sizes[idx], net_factories[idx], outputs[idx])