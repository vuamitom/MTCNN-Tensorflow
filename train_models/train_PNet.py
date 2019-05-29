#coding:utf-8
from train_models.mtcnn_model import P_Net
from train_models.train import train


def train_PNet(base_dir, prefix, log_dir, model_checkpoint, end_epoch, display, lr, optimizer='momentum'):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch: max epoch for training
    :param display:
    :param lr: learning rate
    :return:
    """
    net_factory = P_Net
    train(net_factory, prefix, end_epoch, base_dir, log_dir, 
            display=display, 
            base_lr=lr,
            ckpt=model_checkpoint,
            optimizer=optimizer)

if __name__ == '__main__':
    #data path
    base_dir = '/home/tamvm/Projects/MTCNN-Tensorflow/data/imglists/PNet'
    # model_name = 'MTCNN'
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with landmark
    model_path = '/home/tamvm/Projects/MTCNN-Tensorflow/data/MTCNN68_model/PNet_landmark/PNet'
    log_dir =   '/home/tamvm/Projects/MTCNN-Tensorflow/logs'
    model_checkpoint = None #'/home/tamvm/Projects/MTCNN-Tensorflow/data/MTCNN_model/PNet_landmark/PNet-18'
    prefix = model_path
    end_epoch = 1000
    display = 200
    lr = 0.001
    train_PNet(base_dir, prefix, log_dir, model_checkpoint, end_epoch, display, lr, 'adam')
