from __future__ import division
import os,time
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from datasets import pre_train,pre_train_valid,atec
from text_utils import TextEncoder
from utils import encode_dataset, iter_data, get_ema_vars, convert_gradient_to_tensor, shape_list, ResultLogger,average_grads,assign_to_gpu,find_trainable_variables
import socket
import math
import joblib
import threading

tf.logging.set_verbosity(tf.logging.INFO)

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
    u = tf.reduce_mean(x, axis=axis, keepdims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
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


def client(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))

        # make sure you have a resolvable hostname
        # send hostname and port
        host = '%s:%s' % (socket.getfqdn(), sock.getsockname()[1] + 1)  # hack to get around TIME_WAIT
        sock.sendall(bytes(host, 'ascii'))
        response = str(sock.recv(1024), 'ascii')
        print('Received: {}'.format(response))

        return host, response.split()


class LM_transformer_pretrain():
    def __init__(self, args):
        globals().update(args.__dict__)
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        # self.ps_hosts = ps_hosts.split(',')
        # self.worker_hosts = worker_hosts.split(',')
        self.logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
        self.text_encoder = TextEncoder(encoder_path)
        self.encoder = self.text_encoder.encoder
        self.n_vocab = len(self.text_encoder.encoder)
        self.encoder['_start_'] = len(self.encoder)
        self.encoder['_delimiter_'] = len(self.encoder)
        self.encoder['_end_'] = len(self.encoder)
        self.clf_token = self.encoder['_end_']
        self.n_special = 3
        self.n_batch_train = n_batch * n_gpu
        self.n_updates_total = n_step*10000

    def encode_to_tfrecords(self,tfrecord_filename, origin_filename,num_shards=10):
        writers = []
        output_files = []
        for shard in range(num_shards):
            output_file = "%s-%.5d-of-%.5d" % (tfrecord_filename, shard, num_shards)
            output_files.append(output_file)
            writers.append(tf.python_io.TFRecordWriter(output_file))

        counter, shard = 0, 0
        for case in open(origin_filename, 'r', encoding='utf-8'):
            if counter > 0 and counter % 100000 == 0:
                tf.logging.info("Generating case %d for %s." % (counter, tfrecord_filename))
            data = encode_dataset(self.text_encoder, pre_train([case.strip()]))[0]
            #sentence = self.transform_roc_tfrecord(data)  # ,length
            sentence = [[self.encoder['_start_']]+data[0][:n_ctx-2]+[self.encoder['_end_']]]
            frame_feature = list(
                map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), sentence[0]))
            sequence_example = tf.train.SequenceExample(
                # context=tf.train.Features(feature={
                #     'length': tf.train.Feature(int64_list=tf.train.Int64List(value=length))}),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sequence': tf.train.FeatureList(feature=frame_feature)
                })
            )
            counter += 1
            writers[shard].write(sequence_example.SerializeToString())
            shard = (shard + 1) % num_shards

        for writer in writers:
            writer.close()

        return output_files

        # with tf.python_io.TFRecordWriter(tfrecord_filename) as f:
        #     for line in open(origin_filename, 'r', encoding='utf-8'):
        #         data = encode_dataset(self.text_encoder, pre_train([line.strip()]))[0]
        #         sentence = self.transform_roc_tfrecord(data) #,length
        #         frame_feature = list(
        #             map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), sentence[0]))
        #         example = tf.train.SequenceExample(
        #             # context=tf.train.Features(feature={
        #             #     'length': tf.train.Feature(int64_list=tf.train.Int64List(value=length))}),
        #             feature_lists=tf.train.FeatureLists(feature_list={
        #                 'sequence': tf.train.FeatureList(feature=frame_feature)
        #             })
        #         )
        #         f.write(example.SerializeToString())
    def single_example_parser(self,serialized_example):
        # context_features = {
        #     "length": tf.FixedLenFeature([], dtype=tf.int64)
        # }

        sequence_features = {
            "sequence": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            #context_features=context_features,
            sequence_features=sequence_features
        )
        #lengths = context_parsed['length']

        sequences = sequence_parsed['sequence']
        return sequences#,lengths
    def batched_data(self,tfrecord_filename, single_example_parser, batch_size,padded_shapes, num_epochs=1,
                     buffer_size=1000):
        padded_shapes=tf.Dimension(None)
        tfrecord_filenames = ["%s-%.5d-of-%.5d" % (tfrecord_filename, shard, 10) for shard in range(10)]
        dataset = tf.data.TFRecordDataset(tfrecord_filenames).map(single_example_parser).padded_batch(batch_size,padded_shapes=padded_shapes).shuffle(buffer_size).repeat(num_epochs)
        return dataset.make_one_shot_iterator().get_next()

    def transform_roc(self, X1):
        n_batch = len(X1)
        xmb = np.zeros((n_batch, 1, n_ctx, 2), dtype=np.int32)
        mmb = np.zeros((n_batch, 1, n_ctx), dtype=np.float32)
        start = self.encoder['_start_']
        end = self.encoder['_end_']
        max_len = n_ctx - 2

        for i, x1 in enumerate(X1):
            x12 = [start] + x1[:max_len] + [end]
            l12 = len(x12)
            xmb[i, 0, :l12, 0] = x12
            mmb[i, 0, :l12] = 1
        xmb[:, :, :, 1] = np.arange(self.n_vocab + self.n_special, self.n_vocab + self.n_special + n_ctx)
        return xmb, mmb
    def transform_roc_tfrecord(self, X1):
        n_batch = len(X1)
        xmb = np.zeros((n_batch, n_ctx), dtype=np.int32)
        length = np.zeros((n_batch), dtype=np.int32)
        start = self.encoder['_start_']
        end = self.encoder['_end_']
        max_len = n_ctx - 2
        for i, x1 in enumerate(X1):
            x12 = [start] + x1[:max_len] + [end]
            l12 = len(x12)
            xmb[i, :l12] = x12
            length[i] = min(l12,n_ctx)
            #mmb[i, 0, :l12] = 1
        #xmb[:, :, :, 1] = np.arange(self.n_vocab + self.n_special, self.n_vocab + self.n_special + n_ctx)
        return xmb#, length

    def model(self, X, M, train=False, reuse=False,num_ps=1):
        with tf.variable_scope('model_lm', reuse=reuse, partitioner=tf.fixed_size_partitioner(num_shards=16)):
            we = tf.get_variable("we", [self.n_vocab + self.n_special + n_ctx, n_embd],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            we = dropout(we, embd_pdrop, train)

            X = tf.reshape(X, [-1, n_ctx, 2])
            M = tf.reshape(M, [-1, n_ctx])

            h = embed(X, we)
            for layer in range(n_layer):
                h = block(h, 'h%d' % layer, train=train, scale=True)

            lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
            lm_logits = tf.matmul(lm_h, we, transpose_b=True)
            lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits,
                                                                       labels=tf.reshape(X[:, 1:, 0], [-1]))
            lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])
            lm_losses = tf.reduce_sum(lm_losses * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)

            return lm_losses

    def ccc_train(self):
        # Resolve hostnames and ports of other nodes
        host, hosts = client(bootstrap_host, bootstrap_port)

        # Create a cluster and identify the job name and task of this node
        cluster = tf.train.ClusterSpec({
            'ps': hosts[:num_ps],
            'worker': hosts[num_ps:]
        })

        task = hosts.index(host)
        job_name = ('ps', 'worker')[task >= num_ps]
        task = cluster.job_tasks(job_name).index(host)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        server = tf.train.Server(cluster, job_name=job_name, task_index=task,config=tf_config)

        if job_name == 'ps':
            # create a shared queue on the parameter server which is visible on /job:ps/task:%d
            with tf.device('/job:ps/task:%d' % task):
                queue = tf.FIFOQueue(cluster.num_tasks('worker'), tf.int32, shared_name='done_queue%d' % task)

            # wait for the queue to be filled
            with tf.Session(server.target) as sess:
                for i in range(cluster.num_tasks('worker')):
                    sess.run(queue.dequeue())
                    print('ps:%d received "done" from worker:%d' % (task, i))
                print('ps:%d quitting' % task)

        elif job_name == 'worker':
            with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % task, cluster=cluster)):
                global_step = tf.train.get_or_create_global_step()

                sentences = self.batched_data(tfrecord_filename, self.single_example_parser, self.n_batch_train,
                                            padded_shapes=tf.Dimension(n_ctx), num_epochs=n_iter)
                sentences = tf.cast(sentences,tf.int32)
                max_len = tf.shape(sentences)[1]#sentences.get_shape()[1]
                xmb = tf.reshape(sentences,[self.n_batch_train, 1, max_len, 1])
                M_train = tf.cast(tf.reshape(tf.sign(xmb), [self.n_batch_train, 1, max_len]),tf.float32)
                positions = tf.reshape(tf.range(self.n_vocab + self.n_special, self.n_vocab + self.n_special + max_len),shape=[1, 1, max_len, 1])
                    #tf.constant(np.arange(self.n_vocab + self.n_special, self.n_vocab + self.n_special + max_len),shape=[1, 1, max_len, 1])
                positions = tf.tile(positions,[self.n_batch_train,1,1,1])
                X_train = tf.concat([xmb,positions],axis=3)

                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2, epsilon=e)
                gpu_grads = []
                gpu_loss = []
                gpu_ppl = []
                xs = [X_train, M_train]
                xs = (tf.split(x, n_gpu, 0) for x in xs)
                for i, xs in enumerate(zip(*xs)):
                    do_reuse = True if i > 0 else None
                    with tf.device(assign_to_gpu(i)), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
                        lm_losses = self.model(*xs, train=True,num_ps=num_ps)
                        train_ppl_single = tf.reduce_mean(math.e**lm_losses)
                        train_loss_single = tf.reduce_mean(lm_losses)
                        gpu_loss.append(train_loss_single)
                        gpu_ppl.append(train_ppl_single)
                        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2, epsilon=e)
                        raw_grads_and_vars = optimizer.compute_gradients(train_loss_single)
                        grads_and_vars = [(tf.clip_by_global_norm([gv[0]], max_grad_norm)[0][0], gv[1]) for gv in
                                          raw_grads_and_vars]
                        gpu_grads.append(grads_and_vars)

                train_ppl = tf.reduce_mean(gpu_ppl)
                train_loss = tf.reduce_mean(gpu_loss)
                grads = average_grads(gpu_grads)

                train_op = optimizer.apply_gradients(grads,
                                                     global_step=global_step)

                saver = tf.train.Saver(max_to_keep=5)

                X = tf.placeholder(tf.int32, [None, 1, n_ctx, 2])
                M = tf.placeholder(tf.float32, [None, 1, n_ctx])
                valid_lm_losses = self.model(X, M, train=False, reuse=True)
                valid_ppl = tf.reduce_mean(math.e**valid_lm_losses)
                valid_loss = tf.reduce_mean(valid_lm_losses)

                self.params = find_trainable_variables('model_lm')
                tf.summary.scalar('train_loss', train_loss)
                #tf.summary.scalar('valid_loss', valid_loss)
                tf.summary.scalar('train_ppl', train_ppl)
                #tf.summary.scalar('valid_ppl', valid_ppl)
                summary_op = tf.summary.merge_all()

            done_ops = []
            # create a shared queue on the worker which is visible on /job:ps/task:%d
            for i in range(cluster.num_tasks('ps')):
                with tf.device('/job:ps/task:%d' % i):
                    with tf.name_scope('done_queue'):
                        done_queue = tf.FIFOQueue(cluster.num_tasks('worker'), tf.int32,
                                                  shared_name='done_queue' + str(i))
                        done_ops.append(done_queue.enqueue(task))
            scaffold = tf.train.Scaffold(saver=saver)
            summary_hook = tf.train.SummarySaverHook(save_steps=1000, output_dir=save_dir,summary_op=summary_op)
            hooks = [summary_hook,# tf.train.CheckpointSaverHook(save_secs=600, checkpoint_dir=save_dir, saver=saver),
                     tf.train.StopAtStepHook(last_step=1000000),
                     tf.train.LoggingTensorHook({'step': global_step, 'train_loss': train_loss, 'ppl':train_ppl}, every_n_iter=100),
                     tf.train.FinalOpsHook([done_ops])]
            valid_data = pre_train_valid(valid_dir)
            vaX1 = encode_dataset(self.text_encoder, pre_train(valid_data))[0]
            vaX, vaM = self.transform_roc(vaX1)
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(task == 0),
                                                   hooks=hooks,
                                                   save_checkpoint_secs=600,
                                                   checkpoint_dir=save_dir,
                                                   scaffold=scaffold) as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                try:
                    while not coord.should_stop():

                        ppl,loss, _, step = sess.run([train_ppl,train_loss, train_op, global_step])#,options=run_options, run_metadata=run_metadata)
                        if step % steps_to_validate == 0:
                            va_cost = []
                            va_ppl = []
                            for xm, mm in iter_data((vaX, vaM), n_batch=self.n_batch_train, truncate=False,
                                                    verbose=True):

                                ps = sess.run(self.params)
                                joblib.dump(ps, save_dir + 'model_lm.params', protocol=2)
                                res,ppl = sess.run([valid_loss,valid_ppl],
                                               {X: xm, M: mm})
                                va_cost.append(np.sum(res))
                                va_ppl.append(np.sum(ppl))

                            va_cost = np.average(va_cost)
                            va_ppl = np.average(va_ppl)
                            tf.logging.info('=========n_steps:\t%d valid_cost:\t%.3f valid ppl:\t%.3f==========' % (step, va_cost,va_ppl))

                except tf.errors.OutOfRangeError:
                    print('Epochs Complete!')
                finally:
                    coord.request_stop()
                coord.join(threads)


class LM_transformer_similar():
    def __init__(self,args):
        globals().update(args.__dict__)
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.text_encoder = TextEncoder(encoder_path)
        self.encoder = self.text_encoder.encoder
        self.n_vocab = len(self.text_encoder.encoder)
        self.n_y = 2
        self.encoder['_start_'] = len(self.encoder)
        self.encoder['_delimiter_'] = len(self.encoder)
        self.encoder['_end_'] = len(self.encoder)
        self.clf_token = self.encoder['_end_']
        self.n_special = 3
        self.n_batch_train = n_batch * n_gpu
        self.n_updates_total = n_iter

    def transform_roc(self,X1, X2):
        n_batch = len(X1)
        xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
        mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
        start = self.encoder['_start_']
        delimiter = self.encoder['_delimiter_']
        max_len = (n_ctx-3)//2

        for i, (x1, x2), in enumerate(zip(X1, X2)):
            x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [self.clf_token]
            x13 = [start] + x2[:max_len] + [delimiter] + x1[:max_len] + [self.clf_token]
            l12 = len(x12)
            l13 = len(x13)
            xmb[i, 0, :l12, 0] = x12
            xmb[i, 1, :l13, 0] = x13
            mmb[i, 0, :l12] = 1
            mmb[i, 1, :l13] = 1
        xmb[:, :, :, 1] = np.arange(self.n_vocab + self.n_special, self.n_vocab + self.n_special + n_ctx)
        return xmb, mmb

    def model(self,X, M, Y, train=False, reuse=False):
        with tf.variable_scope('model_lm', reuse=reuse):
            we = tf.get_variable("we", [self.n_vocab + self.n_special + n_ctx, n_embd],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            we = dropout(we, embd_pdrop, train)

            X = tf.reshape(X, [-1, n_ctx, 2])
            M = tf.reshape(M, [-1, n_ctx])

            h = embed(X, we)
            for layer in range(n_layer):
                h = block(h, 'h%d' % layer, train=train, scale=True)

            lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
            lm_logits = tf.matmul(lm_h, we, transpose_b=True)
            lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits,
                                                                       labels=tf.reshape(X[:, 1:, 0], [-1]))
            lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])
            lm_losses = tf.reduce_sum(lm_losses * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)
        with tf.variable_scope('model_clf',reuse=reuse):
            clf_h = tf.reshape(h, [-1, n_embd])
            pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], self.clf_token), tf.float32), 1), tf.int32)
            clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * (n_ctx) + pool_idx)

            clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
            if train and clf_pdrop > 0:
                shape = shape_list(clf_h)
                shape[1] = 1
                clf_h = tf.nn.dropout(clf_h, 1 - clf_pdrop, shape)
            clf_h = tf.reshape(clf_h, [-1,shape_list(clf_h)[1]*shape_list(clf_h)[2]])
            clf_logits = clf(clf_h, 2, train=train)
            #clf_logits = tf.reshape(clf_logits, [-1, 2])
            clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)

            log_weight = (2.0 * pos_weight -1) * tf.cast(Y,tf.float32) + 1 - pos_weight
            clf_losses = log_weight * clf_losses
            return clf_logits, clf_losses, lm_losses

    def train(self):
        global_step = tf.train.get_or_create_global_step()
        X_train = tf.placeholder(tf.int32, [self.n_batch_train, 2, n_ctx, 2])
        M_train = tf.placeholder(tf.float32, [self.n_batch_train, 2, n_ctx])
        X = tf.placeholder(tf.int32, [None, 2, n_ctx, 2])
        M = tf.placeholder(tf.float32, [None, 2, n_ctx])

        Y_train = tf.placeholder(tf.int32, [self.n_batch_train])
        Y = tf.placeholder(tf.int32, [None])

        #self.train, self.logits, self.clf_losses, self.lm_losses = self.mgpu_train(self.X_train, self.M_train, self.Y_train)

        xs = [X_train,M_train,Y_train]
        gpu_ops = []
        gpu_grads = []
        xs = (tf.split(x, n_gpu, 0) for x in xs)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2, epsilon=e)
        for i, xs in enumerate(zip(*xs)):
            do_reuse = True if i > 0 else None
            with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
                logits, clf_losses, lm_losses = self.model(*xs, train=True, reuse=do_reuse)
                if lm_coef > 0:
                    train_loss = tf.reduce_mean(clf_losses) + lm_coef * tf.reduce_mean(lm_losses)
                else:
                    train_loss = tf.reduce_mean(clf_losses)
                raw_grads_and_vars = optimizer.compute_gradients(train_loss)
                grads_and_vars = [(tf.clip_by_global_norm([gv[0]], max_grad_norm)[0][0], gv[1]) for gv in
                                  raw_grads_and_vars]
                gpu_grads.append(grads_and_vars)
                gpu_ops.append([logits, clf_losses, lm_losses])
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        logits, clf_losses, lm_losses = ops
        grads = average_grads(gpu_grads)

        train_op = optimizer.apply_gradients(grads,
                                             global_step=global_step)
        clf_loss = tf.reduce_mean(clf_losses)
        saver = tf.train.Saver(max_to_keep=5)
        self.params = find_trainable_variables('model_lm')

        self.eval_mgpu_logits, self.eval_mgpu_clf_losses, self.eval_mgpu_lm_losses = self.mgpu_predict(X_train, M_train, Y_train)
        self.eval_logits, self.eval_clf_losses, self.eval_lm_losses = self.model(X, M, Y, train=False, reuse=True)
        self.eval_clf_loss = tf.reduce_mean(self.eval_clf_losses)
        self.eval_mgpu_clf_loss = tf.reduce_mean(self.eval_mgpu_clf_losses)

        summary_op = tf.get_collection(tf.GraphKeys.SUMMARIES)

        def trva_split(data, index):
            return [data[i] for i in index]
        x1, x2, y = encode_dataset(self.text_encoder, atec(data_dir) )

        valid_index = np.load('data/valid_index.npy')
        if data_dir=='data/para.tsv':
            valid_index = np.concatenate([valid_index,valid_index+len(y)//4,valid_index+len(y)//2,valid_index+3*len(y)//4])
        valid_index = valid_index.tolist()
        train_index = list(set(valid_index) ^ set(range(len(y))))
        trX1, trX2, trY = trva_split(x1, train_index), trva_split(x2, train_index), trva_split(y, train_index)
        vaX1, vaX2, vaY = trva_split(x1, valid_index), trva_split(x2, valid_index), trva_split(y, valid_index)
        trX, trM = self.transform_roc(trX1, trX2)
        vaX, vaM = self.transform_roc(vaX1, vaX2)

        n_train = len(trY)
        n_valid = len(vaY)
        self.n_updates_total = (n_train // self.n_batch_train) * n_iter

        n_updates = 0
        n_epochs = 0

        def log():
            def iter_apply(Xs, Ms, Ys):
                fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
                results = []
                for xmb, mmb, ymb in iter_data((Xs, Ms, Ys), n_batch=self.n_batch_train, truncate=False, verbose=True):
                    n = len(xmb)
                    if n == self.n_batch_train:
                        res = sess.run([self.eval_mgpu_logits, self.eval_mgpu_clf_loss],
                                            {X_train: xmb, M_train: mmb, Y_train: ymb})
                    else:
                        res = sess.run([self.eval_logits, self.eval_clf_loss],
                                            {X: xmb, M: mmb, Y: ymb})
                    res = [r * n for r in res]
                    results.append(res)
                results = zip(*results)
                return [fn(res) for res, fn in zip(results, fns)]
            # global best_score
            tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
            va_logits, va_cost = iter_apply(vaX, vaM, vaY)
            tr_cost = tr_cost / len(trY[:n_valid])
            va_cost = va_cost / n_valid
            tr_f1 = f1_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
            va_f1 = f1_score(vaY, np.argmax(va_logits, 1)) * 100.
            tf.logging.info('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_f1, va_f1))

        scaffold = tf.train.Scaffold(saver=saver)
        summary_hook = tf.train.SummarySaverHook(save_steps=1000, output_dir=save_dir,summary_op=summary_op)
        hooks = [summary_hook]
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.train.MonitoredTrainingSession(hooks=hooks,
                                               save_checkpoint_secs=600,
                                               checkpoint_dir=save_dir,
                                               scaffold=scaffold,
                                               config=tf_config) as sess:
            if preload:
                sess.run(
                    [p.assign(ip) for p, ip in
                     zip(self.params, joblib.load(save_dir+ 'model_lm.params'))])

            for i in range(n_iter):
                for xmb, mmb, ymb in iter_data((shuffle(trX, trM, trY, random_state=np.random)), n_batch=self.n_batch_train, truncate=True, verbose=True):
                    cost, _ = sess.run([clf_loss, train_op], {X_train:xmb, M_train:mmb, Y_train:ymb})
                    n_updates += 1
                    if n_updates%1000==0 :
                        log()
                n_epochs += 1
                log()

            teX1, teX2, _ = encode_dataset(self.text_encoder, atec(data_dir))
            teX, teM = self.transform_roc(teX1, teX2)

            pred_fn = lambda x: np.argmax(x, 1)
            logits = []
            for xmb, mmb in iter_data((teX, teM), n_batch=self.n_batch_train, truncate=False, verbose=True):
                n = len(xmb)
                if n == self.n_batch_train:
                    logits.append(sess.run(self.eval_mgpu_logits, {X_train: xmb, M_train: mmb}))
                else:
                    logits.append(sess.run(self.eval_logits, {X: xmb, M: mmb}))
            logits = np.concatenate(logits, 0)
            predictions = pred_fn(logits)
            return predictions

    def mgpu_predict(self,*xs):
        gpu_ops = []
        xs = (tf.split(x, n_gpu, 0) for x in xs)
        for i, xs in enumerate(zip(*xs)):
            with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
                clf_logits, clf_losses, lm_losses = self.model(*xs, train=False, reuse=True)
                gpu_ops.append([clf_logits, clf_losses, lm_losses])
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        return ops

