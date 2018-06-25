import os
import math
import json
from sklearn.externals import joblib
import random
import numpy as np
import tensorflow as tf

from functools import partial
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from datasets import atec
from text_utils import TextEncoder
from utils import encode_dataset, flatten, iter_data, find_trainable_variables, get_ema_vars, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path

# seed = 42
# n_iter = 3
# n_batch = 8
# max_grad_norm = 1
# lr = 6.25e-5
# lr_warmup = 0.002
# max_len = 27
# n_gpu = 4
# opt = 'adam'
# afn = 'gelu'
# n_embd = 384
# n_head = 6
# n_layer = 3
# embd_pdrop = 0.1
# attn_pdrop = 0.1
# resid_pdrop = 0.1
# clf_pdrop = 0.1
# l2 = 0.01
# vector_l2 = True
# lr_schedule = 'warmup_linear'
# encoder_path = 'data/vocab.txt'
# desc = 'tmp'
# log_dir = 'log/'
# save_dir = 'save/'
# data_dir = 'data/AB_unk.tsv'
# n_transfer = 12
# lm_coef = 0
# b1 = 0.9
# b2 = 0.999
# e = 1e-8
# pre_load = False


def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b


class LM_transformer():
    def __init__(self,args):
        globals().update(args.__dict__)
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
        self.text_encoder = TextEncoder(encoder_path)
        self.encoder = self.text_encoder.encoder
        self.n_vocab = len(self.text_encoder.encoder)
        self.n_y = 2
        self.encoder['_start_'] = len(self.encoder)
        self.encoder['_end_'] = len(self.encoder)
        self.clf_token = self.encoder['_end_']
        self.n_special = 2
        self.n_batch_train = n_batch * n_gpu


    def transform_roc(self,X1, X2):
        n_batch = len(X1)
        xmb = np.zeros((n_batch, 2, 2+max_len, 2), dtype=np.int32)
        mmb = np.zeros((n_batch, 2, 2+max_len), dtype=np.float32)
        start = self.encoder['_start_']
        for i, (x1, x2), in enumerate(zip(X1, X2)):
            x12 = [start] + x1[:max_len] + [self.clf_token]
            x13 = [start] + x2[:max_len] + [self.clf_token]
            l12 = len(x12)
            l13 = len(x13)
            xmb[i, 0, :l12, 0] = x12
            xmb[i, 1, :l13, 0] = x13
            mmb[i, 0, :l12] = 1
            mmb[i, 1, :l13] = 1
        xmb[:, :, :, 1] = np.arange(self.n_vocab + self.n_special, self.n_vocab + self.n_special + max_len+2)
        return xmb, mmb

    def build_graph(self,tr=True):
        self.X_train = tf.placeholder(tf.int32, [self.n_batch_train, 2, max_len+2, 2])
        self.M_train = tf.placeholder(tf.float32, [self.n_batch_train, 2, max_len+2])
        self.X = tf.placeholder(tf.int32, [None, 2, max_len+2, 2])
        self.M = tf.placeholder(tf.float32, [None, 2, max_len+2])

        self.Y_train = tf.placeholder(tf.int32, [self.n_batch_train])
        self.Y = tf.placeholder(tf.int32, [None])
        if tr:
            self.train, self.logits, self.clf_losses, self.lm_losses = self.mgpu_train(self.X_train, self.M_train, self.Y_train)
            self.clf_loss = tf.reduce_mean(self.clf_losses)

        self.params = find_trainable_variables('model')
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

        self.eval_mgpu_logits, self.eval_mgpu_clf_losses, self.eval_mgpu_lm_losses = self.mgpu_predict(self.X_train, self.M_train, self.Y_train)
        self.eval_logits, self.eval_clf_losses, self.eval_lm_losses = self.model(self.X,self.M, self.Y, train=False, reuse=True)
        self.eval_clf_loss = tf.reduce_mean(self.eval_clf_losses)
        self.eval_mgpu_clf_loss = tf.reduce_mean(self.eval_mgpu_clf_losses)

    def mgpu_train(self,*xs):
        gpu_ops = []
        gpu_grads = []
        xs = (tf.split(x, n_gpu, 0) for x in xs)
        for i, xs in enumerate(zip(*xs)):
            do_reuse = True if i > 0 else None
            with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
                clf_logits, clf_losses, lm_losses = self.model(*xs, train=True, reuse=do_reuse)
                if lm_coef > 0:
                    train_loss = tf.reduce_mean(clf_losses) + lm_coef * tf.reduce_mean(lm_losses)
                else:
                    train_loss = tf.reduce_mean(clf_losses)
                params = find_trainable_variables("model")
                grads = tf.gradients(train_loss, params)
                grads = list(zip(grads, params))
                gpu_grads.append(grads)
                gpu_ops.append([clf_logits, clf_losses, lm_losses])
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        grads = average_grads(gpu_grads)
        grads = [g for g, p in grads]
        train = opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup),
                             self.n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1,
                             b2=b2, e=e)
        return [train] + ops

    def model(self,X, M, Y, train=False, reuse=False):
        with tf.variable_scope('model', reuse=reuse):
            we = tf.get_variable("we", [self.n_vocab + self.n_special + max_len, n_embd],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            we = dropout(we, embd_pdrop, train)

            X = tf.reshape(X, [-1, max_len+2, 2])
            M = tf.reshape(M, [-1, max_len+2])

            h = embed(X, we)
            for layer in range(n_layer):
                h = block(h, 'h%d' % layer, train=train, scale=True)

            lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
            lm_logits = tf.matmul(lm_h, we, transpose_b=True)
            lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits,
                                                                       labels=tf.reshape(X[:, 1:, 0], [-1]))
            lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])
            lm_losses = tf.reduce_sum(lm_losses * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)

            clf_h = tf.reshape(h, [-1, n_embd])
            pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], self.clf_token), tf.float32), 1), tf.int32)
            clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * (max_len+2) + pool_idx)

            clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
            if train and clf_pdrop > 0:
                shape = shape_list(clf_h)
                shape[1] = 1
                clf_h = tf.nn.dropout(clf_h, 1 - clf_pdrop, shape)
            clf_h = tf.reshape(clf_h, [-1, n_embd])
            clf_logits = clf(clf_h, 1, train=train)
            clf_logits = tf.reshape(clf_logits, [-1, 2])

            clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
            return clf_logits, clf_losses, lm_losses


    def iter_apply(self,Xs, Ms, Ys):
        fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
        results = []
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=self.n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            if n == self.n_batch_train:
                res = self.sess.run([self.eval_mgpu_logits, self.eval_mgpu_clf_loss], {self.X_train: xmb, self.M_train: mmb, self.Y_train: ymb})
            else:
                res = self.sess.run([self.eval_logits, self.eval_clf_loss], {self.X: xmb, self.M: mmb, self.Y: ymb})
            res = [r * n for r in res]
            results.append(res)
        results = zip(*results)
        return [fn(res) for res, fn in zip(results, fns)]

    def train(self,data_dir='data/AB_unk.tsv'):
        def trva_split(data, index):
            return [data[i] for i in index]
        self.x1, self.x2, self.y = encode_dataset(atec(data_dir), encoder=self.text_encoder)

        valid_index = np.load('data/valid_index.npy').tolist()
        train_index = list(set(valid_index) ^ set(range(len(self.y))))
        trX1, trX2, trY = trva_split(self.x1, train_index), trva_split(self.x2, train_index), trva_split(self.y, train_index)
        vaX1, vaX2, vaY = trva_split(self.x1, valid_index), trva_split(self.x2, valid_index), trva_split(self.y, valid_index)
        trX, trM = self.transform_roc(trX1, trX2)
        vaX, vaM = self.transform_roc(vaX1, vaX2)

        n_train = len(trY)
        n_valid = len(vaY)
        self.n_updates_total = (n_train // self.n_batch_train) * n_iter
        self.build_graph()
        if pre_load:
            shapes = json.load(open('model/params_shapes.json'))
            offsets = np.cumsum([np.prod(shape) for shape in shapes])
            init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
            init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
            init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
            init_params[0] = init_params[0][:max_len+2]
            init_params[0] = np.concatenate([init_params[1], (np.random.randn(self.n_special, n_embd)*0.02).astype(np.float32), init_params[0]], 0)
            del init_params[1]

            if self.n_transfer == -1:
                self.n_transfer = 0
            else:
                self.n_transfer = 1+self.n_transfer*12
            self.sess.run([p.assign(ip) for p, ip in zip(self.params[:self.n_transfer], init_params[:self.n_transfer])])

        n_updates = 0
        n_epochs = 0
        self.save(os.path.join(save_dir, desc, 'best_params.jl'))
        best_score = 0

        def log():
            global best_score
            tr_logits, tr_cost = self.iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
            va_logits, va_cost = self.iter_apply(vaX, vaM, vaY)
            tr_cost = tr_cost / len(trY[:n_valid])
            va_cost = va_cost / n_valid
            tr_f1 = f1_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
            va_f1 = f1_score(vaY, np.argmax(va_logits, 1)) * 100.
            self.logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_f1,
                       va_acc=va_f1)
            print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_f1, va_f1))
            score = va_f1
            if score > best_score:
                best_score = score
                self.save(os.path.join(save_dir, desc, 'best_params.jl'))

        for i in range(n_iter):
            for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trY, random_state=np.random), n_batch=self.n_batch_train, truncate=True, verbose=True):
                cost, _ = self.sess.run([self.clf_loss, self.train], {self.X_train:xmb, self.M_train:mmb, self.Y_train:ymb})
                n_updates += 1
                if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
                    log()
            n_epochs += 1
            log()

    def iter_predict(self,Xs, Ms):
        logits = []
        for xmb, mmb in iter_data(Xs, Ms, n_batch=self.n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            if n == self.n_batch_train:
                logits.append(self.sess.run(self.eval_mgpu_logits, {self.X_train: xmb, self.M_train: mmb}))
            else:
                logits.append(self.sess.run(self.eval_logits, {self.X: xmb, self.M: mmb}))
        logits = np.concatenate(logits, 0)
        return logits

    def mgpu_predict(self,*xs):
        gpu_ops = []
        xs = (tf.split(x, n_gpu, 0) for x in xs)
        for i, xs in enumerate(zip(*xs)):
            with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
                clf_logits, clf_losses, lm_losses = self.model(*xs, train=False, reuse=True)
                gpu_ops.append([clf_logits, clf_losses, lm_losses])
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        return ops

    def predict(self,data_dir):
        teX1, teX2, _ = encode_dataset(atec(data_dir), encoder=self.text_encoder)
        teX, teM = self.transform_roc(teX1, teX2)
        self.build_graph(tr=False)
        self.sess.run(
            [p.assign(ip) for p, ip in zip(self.params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
        pred_fn = lambda x: np.argmax(x, 1)
        predictions = pred_fn(self.iter_predict(teX, teM))
        return predictions

    def save(self,path):
        ps = self.sess.run(self.params)
        joblib.dump(ps, make_path(path))