#!/bin/env python
# encoding:utf-8

from util import *
from model import *
from sklearn.metrics import roc_auc_score
import progressbar

class FNN(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        node_in = num_inputs * embed_size
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            l = xw

            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                print(l.shape, wi.shape, bi.shape)
                l = tf.nn.dropout(
                    activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
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

    train_size = train_data[0].shape[0]
    test_size = test_data[0].shape[0]
    num_feas = len(FIELD_SIZES)

    min_round = 1
    num_round = 200
    early_stop_round = 5
    batch_size = 1024

    field_sizes = FIELD_SIZES
    field_offsets = FIELD_OFFSETS

    train_data = split_data(train_data)
    test_data = split_data(test_data)
    tmp = []
    for x in field_sizes:
        if x > 0:
            tmp.append(x)
    field_sizes = tmp
    print('remove empty fields', field_sizes)
        
    fnn_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0
    }
    print(fnn_params)
    model = FNN(**fnn_params)

    def train(model):
        history_score = []
        for i in range(num_round):
            fetches = [model.optimizer, model.loss]
            if batch_size > 0:
                ls = []
                bar = progressbar.ProgressBar()
                print('[%d]\ttraining...' % i)
                for j in bar(range(int(train_size / batch_size + 1))):
                    X_i, y_i = slice(train_data, j * batch_size, batch_size)
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