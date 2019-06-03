import sys
sys.path.append('../..')
import tensorflow as tf
import os 
from train_models.mtcnn_model import P_Net, R_Net, O_Net

def save_eval_graph(net_factory, data_size, path):
    img = tf.placeholder(tf.float32, shape=[1, data_size, data_size, 3], name='input_image')
    cls_prob, bbox_pred, landmark_pred = net_factory(img, training=False)
    print('cls_prob', cls_prob, cls_prob.op.name)
    
    g = tf.get_default_graph()
    tf.contrib.quantize.create_eval_graph(input_graph=g)
    for op in g.get_operations(): 
        print(op.name, op)
    basedir = os.path.dirname(path)
    eval_graph_file = os.path.join(basedir, 'eval_graph_def_'+ str(data_size) + '.pb')
    checkpoint_name = os.path.join(basedir, 'eval_graph_checkpoint_'+ str(data_size) + '.ckpt')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                        gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, path)
        with open(eval_graph_file, 'w') as f:
            f.write(str(g.as_graph_def()))
        saver = tf.train.Saver()
        saver.save(sess, checkpoint_name)
    tf.reset_default_graph()
    return eval_graph_file, checkpoint_name


def export_model(path, data_size, net_factory, output_name, quantized=False):
    # if data_size is None:
    #     return
    print ('exporting model ', path, 'data_size=', data_size)
    # eval_graph_file, checkpoint_name = save_eval_graph(net_factory, data_size, path)
    # if True:
    #     print('eval_graph_file ', eval_graph_file)
    #     print('checkpoint_name', checkpoint_name)
    #     return 
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                        gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # inputs, outputs = None, None
            # if data_size is None:
            #     img = tf.placeholder(tf.float32, name='input_image')
            #     width = tf.placeholder(tf.float32, name='image_width')
            #     height = tf.placeholder(tf.float32, name='image_height')
            #     image_reshape = tf.reshape(img, [1, height, width, 3])
            #     inputs = [img, width, height]
            #     cls_prob, bbox_pred, landmark_pred = net_factory(imdata_sizeage_reshape, training=False)
            #     outputs = [cls_prob, bbox_pred, landmark_pred]
            # else:
            img = tf.placeholder(tf.float32, shape=[1, data_size, data_size, 3], name='input_image')
            cls_prob, bbox_pred, landmark_pred = net_factory(img, training=False)
            inputs = [img]
            outputs = [cls_prob, bbox_pred, landmark_pred]
            g = tf.get_default_graph()
            tf.contrib.quantize.create_eval_graph(input_graph=g)
            saver = tf.train.Saver()
            saver.restore(sess, path)
            converter = tf.lite.TFLiteConverter.from_session(sess, inputs, outputs)
            # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            if quantized:
                # converter.inference_type = tf.uint8
                converter.allow_custom_ops = False
                # converter.quantized_input_stats = {}
                # converter.quantized_input_stats['input_image'] = (127.5, 128) # (mean, std)
                # converter.default_ranges_min = 0
                # converter.default_ranges_max = 128
                converter.post_training_quantize = True
            tflite_model = converter.convert()
            op = os.path.join(os.path.dirname(path), output_name + '.tflite')
            with open(op, 'wb') as f:
                f.write(tflite_model)
            print('write to ', op)

prefix = ['MTCNN68_model/RNet_landmark/RNet', 'MTCNN68_model/ONet_landmark/ONet']
epoch = [666999, 16]
data_sizes = [24, 48]
net_factories = [R_Net, O_Net]
model_paths = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
outputs = ['RNet', 'ONet']
expect_input_size = 80
quantize_model = True
# get necessary pnet size 
def gen_pnet_sizes(init_bitmap_size):
    pnet_size = 12
    scale_factor = 0.5
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

for s in gen_pnet_sizes(expect_input_size):
    data_sizes.append(s)
    net_factories.append(P_Net)
    model_paths.append('MTCNN68_model/PNet_landmark/PNet-433599')
    outputs.append('PNet_%s' % (s,))

print(model_paths)
for idx in range(0, len(model_paths)):
    if model_paths[idx].find('RNet') >= 0:
        export_model(model_paths[idx], data_sizes[idx], net_factories[idx], outputs[idx], quantize_model)