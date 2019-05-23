#coding:utf-8
import os
import random
import sys
import time

import tensorflow as tf

from prepare_data.tfrecord_utils import sample_to_tfrecord


# def _add_to_tfrecord(filename, image_example, tfrecord_writer):
#     """Loads data from image and annotations files and add them to a TFRecord.

#     Args:
#       filename: Dataset directory;
#       name: Image name to add to the TFRecord;
#       tfrecord_writer: The TFRecord writer to use for writing.
#     """

#     # image_data, height, width = _process_image_withoutcoder(filename)
#     # example = _convert_to_example_simple(image_example, image_data)
#     sample_to_tfrecord(filename, image_example)
#     tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, net):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    return '%s/train_PNet_landmark.tfrecord' % (output_dir)
    

def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    #tfrecord name 
    tf_filename = _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    # print('1')
    dataset = get_dataset(dataset_dir, net=net)
    # print('2')
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i+1) % 100 == 0:
                sys.stdout.write('\r>> %d/%d images has been converted' % (i+1, len(dataset)))
                #sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
                sys.stdout.flush()
            filename = image_example['filename']
            # print('landmark size = ', len(image_example['bbox']['landmarks']))
            example = sample_to_tfrecord(filename, image_example)
            tfrecord_writer.write(example.SerializeToString())
            # _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dataset_dir, net='PNet'):
    #get file name , label and anotation
    #item = 'imglists/PNet/train_%s_raw.txt' % net
    item = 'imglists/PNet/train_%s_landmark.txt' % net
    no_landmarks = 68
    dataset_txt = os.path.join(dataset_dir, item)
    #print(dataset_dir)
    imagelist = open(dataset_txt, 'r')

    dataset = []
    all_lines = imagelist.readlines()
    print('all lines = ', len(all_lines))
    for line in all_lines:
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        #print(data_example['filename'])
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['landmarks'] = [0.0 for x in range(0, no_landmarks*2)]
     
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        elif len(info) == 138:
            # print('info len = ', len(info))
            bbox['landmarks'] = [float(x) for x in info[2:(no_landmarks * 2 + 2)]]
            # bbox['xlefteye'] = float(info[2])
            # bbox['ylefteye'] = float(info[3])
            # bbox['xrighteye'] = float(info[4])
            # bbox['yrighteye'] = float(info[5])
            # bbox['xnose'] = float(info[6])
            # bbox['ynose'] = float(info[7])
            # bbox['xleftmouth'] = float(info[8])
            # bbox['yleftmouth'] = float(info[9])
            # bbox['xrightmouth'] = float(info[10])
            # bbox['yrightmouth'] = float(info[11])            
        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset


if __name__ == '__main__':
    datadir = '/home/tamvm/Projects/MTCNN-Tensorflow/data/'
    net = 'PNet'
    output_directory = '/home/tamvm/Projects/MTCNN-Tensorflow/data/imglists/PNet'
    run(datadir, net, output_directory, shuffling=True)
