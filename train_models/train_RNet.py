#coding:utf-8
from train_models.mtcnn_model import R_Net
from train_models.train import train


def train_RNet(base_dir, prefix, log_dir, model_checkpoint, end_epoch, display, lr, optimizer='momentum'):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = R_Net
    train(net_factory, 
            prefix, end_epoch, 
            base_dir, log_dir,
            display=display, base_lr=lr,
            ckpt=model_checkpoint,
            optimizer=optimizer)

if __name__ == '__main__':
    base_dir = '/home/tamvm/Projects/MTCNN-Tensorflow/data/imglists_noLM/RNet'

    model_name = 'MTCNN68'
    model_path = '/home/tamvm/Projects/MTCNN-Tensorflow/data/%s_model/RNet_No_Landmark/RNet' % model_name
    prefix = model_path
    log_dir =   '/home/tamvm/Projects/MTCNN-Tensorflow/logs'
    end_epoch = 1000
    display = 100
    model_checkpoint = None
    lr = 0.001
    train_RNet(base_dir, prefix, log_dir, model_checkpoint, end_epoch, display, lr, 'adam')