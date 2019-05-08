#!/bin/env python
# encoding:utf-8

from util import *
from model import *
from sklearn.metrics import roc_auc_score
import progressbar

class FM(Model):
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        Model.__init__(self)
        # 一次、二次交叉、偏置项
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('v', [input_dim, factor_order], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']
            
            # [(x1+x2+x3)^2 - (x1^2+x2^2+x3^2)]/2
            # 先计算所有的交叉项，再减去平方项(自己和自己相乘)
            # y = w0 + W * X + <vi, vj> xi xj
            # https://zhuanlan.zhihu.com/p/37963267
            X_square = tf.SparseTensor(
                self.X.indices, 
                tf.square(self.X.values), 
                tf.to_int64(tf.shape(self.X))
            ) # X 中每一项计算平方, N * INPUT_DIM 的稀疏矩阵
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v)) # X v矩阵先相乘后平方, N*k 
            p = 0.5 * tf.reshape(
                tf.reduce_sum(
                    xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(v)), 
                    1
                ),
                [-1, output_dim]
            )
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b + p, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

            #GPU设定
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            # 图中所有variable初始化
            tf.global_variables_initializer().run(session=self.sess)

if __name__ == '__main__':
    train_file = './data/train.txt'
    test_file = './data/test.txt'

    input_dim = INPUT_DIM
    train_data = read_data(train_file)
    test_data = read_data(test_file)
    # train_data = pkl.load(open('./data/train.pkl', 'rb'))
    train_data = shuffle(train_data)
    # test_data = pkl.load(open('./data/test.pkl', 'rb'))

    if train_data[1].ndim > 1:
        print('label must be 1-dim')
        exit(0)
    print('read finish')
    print('train data size:', train_data[0].shape)
    print('test data size:', test_data[0].shape)

    # 训练集与测试集
    train_size = train_data[0].shape[0]
    test_size = test_data[0].shape[0]
    num_feas = len(FIELD_SIZES)

    # 超参数设定
    min_round = 1
    num_round = 200
    early_stop_round = 5
    batch_size = 1024

    field_sizes = FIELD_SIZES
    field_offsets = FIELD_OFFSETS

    # FM参数设定
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }
    print(fm_params)
    model = FM(**fm_params)
    print("training FM...")

    def train(model):
        history_score = []
        for i in range(num_round):
            # 同样是优化器和损失两个op
            fetches = [model.optimizer, model.loss]
            if batch_size > 0:
                ls = []
                bar = progressbar.ProgressBar()
                print('[%d]\ttraining...' % i)
                for j in bar(range(int(train_size / batch_size + 1))):
                    X_i, y_i = slice(train_data, j * batch_size, batch_size)
                    # 训练
                    _, l = model.run(fetches, X_i, y_i)
                    ls.append(l)
            elif batch_size == -1:
                X_i, y_i = slice(train_data)
                _, l = model.run(fetches, X_i, y_i)
                ls = [l]
            train_preds = []
            print('[%d]\tevaluating...' % i)
            bar = progressbar.ProgressBar()
            for j in bar(range(int(train_size / 10000 + 1))):
                X_i, _ = slice(train_data, j * 10000, 10000)
                preds = model.run(model.y_prob, X_i, mode='test')
                train_preds.extend(preds)
            test_preds = []
            bar = progressbar.ProgressBar()
            for j in bar(range(int(test_size / 10000 + 1))):
                X_i, _ = slice(test_data, j * 10000, 10000)
                preds = model.run(model.y_prob, X_i, mode='test')
                test_preds.extend(preds)
            train_score = roc_auc_score(train_data[1], train_preds)
            test_score = roc_auc_score(test_data[1], test_preds)
            print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
            history_score.append(test_score)
            if i > min_round and i > early_stop_round:
                if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                            -1 * early_stop_round] < 1e-5:
                    print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                        np.argmax(history_score), np.max(history_score)))
                    break

    train(model)