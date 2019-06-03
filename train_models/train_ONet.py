#coding:utf-8
from train_models.mtcnn_model import O_Net
from train_models.train import train


def train_ONet(base_dir, prefix, log_dir, model_checkpoint, end_epoch, display, lr, optimizer='momentum'):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = O_Net
    train(net_factory, prefix, end_epoch, base_dir, log_dir, 
            display=display, 
            base_lr=lr,
            ckpt=model_checkpoint,
            optimizer=optimizer)

if __name__ == '__main__':
    base_dir = '/home/tamvm/Projects/MTCNN-Tensorflow/data/imglists/ONet'
    model_path = '/home/tamvm/Projects/MTCNN-Tensorflow/data/MTCNN68_model/ONet_landmark/ONet'
    log_dir =   '/home/tamvm/Projects/MTCNN-Tensorflow/logs'
    prefix = model_path
    end_epoch = 1000
    display = 10
    lr = 0.001
    model_checkpoint = None
    train_ONet(base_dir, prefix, log_dir, model_checkpoint, end_epoch, display, lr, 'adam')

