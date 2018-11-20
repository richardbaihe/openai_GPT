import argparse,os
from process import preprocess
import numpy as np
import tensorflow as tf
from datasets import pre_train,pre_train_valid
from utils import encode_dataset, iter_data,find_trainable_variables
import socket
import math
import joblib
import threading
from model import LM_transformer_pretrain
from model import client

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags

flags.DEFINE_bool('preprocess', False, 'pre process for tfrecord files')
flags.DEFINE_string('save_dir', 'save', 'save path')
flags.DEFINE_string('checkpoint_name', 'pretrain_word', 'checkpoint folder name')
flags.DEFINE_string('data_dir', 'data', 'data path')
flags.DEFINE_string('raw_data_name', 'raw.word.toy', 'raw data name')
# assign a value later
flags.DEFINE_string('raw_data_path',' ','assign a value later')
flags.DEFINE_string('train_data_path',' ','assign a value later ')
flags.DEFINE_string('valid_data_path',' ','assign a value later ')
flags.DEFINE_string('vocab_path',' ','assign a value later ')
flags.DEFINE_string('tfrecord_filename',' ','assign a value later ')
flags.DEFINE_string('model_path',' ','assign a value later ')
flags.DEFINE_string('lm_path',' ','assign a value later ')

flags.DEFINE_string('char_word','word','char or word, which is needed in the preprocess function')
flags.DEFINE_integer('steps_to_validate', 1000, 'validating every n steps')
flags.DEFINE_integer('n_iter', 30, 'total epochs')
flags.DEFINE_integer('n_step', 1000, 'total steps')
flags.DEFINE_integer('n_batch', 8, 'batch size')
flags.DEFINE_integer('n_vocab', 10000, 'vocab size')
flags.DEFINE_integer('n_ctx', 200, 'max length of each sentences')
flags.DEFINE_integer('seed', 42, 'random seed')
flags.DEFINE_integer('max_grad_norm', 1, 'max grad norm')
flags.DEFINE_string('opt', 'adam', 'gradient updating method')
flags.DEFINE_float('b1', 0.9,'adam')
flags.DEFINE_float('b2', 0.999,'adam')
flags.DEFINE_float('e', 1e-8, 'adam')
flags.DEFINE_float('lr', 6.25e-5, 'learning rate')
flags.DEFINE_string('lr_schedule', 'warmup_linear', 'warm up schedule')
flags.DEFINE_float('lr_warmup', 0.002, 'warm up')
flags.DEFINE_bool('pre_load', True, 'pre load or not')
flags.DEFINE_integer('n_gpu', 1, 'nums of gpu used')
flags.DEFINE_integer('n_transfer', 12,'')
flags.DEFINE_float('lm_coef', 0.2, 'language model weight in multi-task training')
flags.DEFINE_string('bootstrap_host', 'localhost','The hostname or IP of the bootstrapping server')
flags.DEFINE_integer('bootstrap_port', '22', 'The port of the bootstrapping server')
flags.DEFINE_integer('num_ps', 1, 'number of parameter server')

args = flags.FLAGS

args.raw_data_path = os.path.join(args.data_dir, args.raw_data_name)
args.train_data_path = os.path.join(args.data_dir, args.checkpoint_name, 'train.' + args.char_word)
args.valid_data_path = os.path.join(args.data_dir, args.checkpoint_name, 'dev.' + args.char_word)
args.vocab_path = os.path.join(args.data_dir, args.checkpoint_name, 'vocab.' + args.char_word)
args.tfrecord_filename = os.path.join(args.data_dir, args.checkpoint_name, 'tfrecord')

args.model_path = os.path.join(args.save_dir , args.checkpoint_name)
args.lm_path = os.path.join(args.save_dir , args.checkpoint_name )


def ccc_train(model,args):
    # Resolve hostnames and ports of other nodes
    host, hosts = client(args.bootstrap_host, args.bootstrap_port)

    # Create a cluster and identify the job name and task of this node
    cluster = tf.train.ClusterSpec({
        'ps': hosts[:args.num_ps],
        'worker': hosts[args.num_ps:]
    })

    task = hosts.index(host)
    job_name = ('ps', 'worker')[task >= args.num_ps]
    task = cluster.job_tasks(job_name).index(host)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name=job_name, task_index=task, config=tf_config)

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
            # with tf.device(tf.train.replica_device_setter()):
            global_step = tf.train.get_or_create_global_step()

            sentences = model.batched_data(args.tfrecord_filename, model.single_example_parser, model.n_batch_train,
                                          padded_shapes=tf.Dimension(args.n_ctx), num_epochs=args.n_iter)
            sentences = tf.cast(sentences, tf.int32)
            max_len = tf.shape(sentences)[1]  # sentences.get_shape()[1]
            xmb = tf.reshape(sentences, [model.n_batch_train, 1, max_len, 1])
            M_train = tf.cast(tf.reshape(tf.sign(xmb), [model.n_batch_train, 1, max_len]), tf.float32)
            positions = tf.reshape(tf.range(model.n_vocab + model.n_special, model.n_vocab + model.n_special + max_len),
                                   shape=[1, 1, max_len, 1])
            positions = tf.tile(positions, [model.n_batch_train, 1, 1, 1])
            X_train = tf.concat([xmb, positions], axis=3)

            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.b1, beta2=args.b2, epsilon=args.e)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                lm_losses = model.model(X_train, M_train, train=True, num_ps=args.num_ps, max_len=max_len)
                train_ppl_single = tf.reduce_mean(math.e ** lm_losses)
                train_loss_single = tf.reduce_mean(lm_losses)
                raw_grads_and_vars = optimizer.compute_gradients(train_loss_single)
                grads_and_vars = [(tf.clip_by_global_norm([gv[0]], args.max_grad_norm)[0][0], gv[1]) for gv in
                                  raw_grads_and_vars]

            train_ppl = train_ppl_single
            train_loss = train_loss_single
            grads = grads_and_vars

            train_op = optimizer.apply_gradients(grads,
                                                 global_step=global_step)

            saver = tf.train.Saver(max_to_keep=5)

            X = tf.placeholder(tf.int32, [None, 1, args.n_ctx, 2])
            M = tf.placeholder(tf.float32, [None, 1, args.n_ctx])
            valid_lm_losses = model.model(X, M, train=False, reuse=True, max_len=args.n_ctx)
            valid_ppl = tf.reduce_mean(math.e ** valid_lm_losses)
            valid_loss = tf.reduce_mean(valid_lm_losses)

            model.params = find_trainable_variables('model_lm')
            tf.summary.scalar('train_loss', train_loss)
            # tf.summary.scalar('valid_loss', valid_loss)
            tf.summary.scalar('train_ppl', train_ppl)
            # tf.summary.scalar('valid_ppl', valid_ppl)
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
        summary_hook = tf.train.SummarySaverHook(save_steps=1000, output_dir=args.model_path, summary_op=summary_op)
        hooks = [summary_hook,
                 tf.train.StopAtStepHook(last_step=model.n_updates_total),
                 tf.train.LoggingTensorHook({'step': global_step, 'train_loss': train_loss, 'ppl': train_ppl},
                                            every_n_iter=10),
                 tf.train.FinalOpsHook([done_ops])]
        valid_data = pre_train_valid(args.valid_data_path)
        vaX1 = encode_dataset(model.text_encoder, pre_train(valid_data))[0]
        vaX, vaM = model.transform_roc(vaX1)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task == 0),
                                               hooks=hooks,
                                               save_checkpoint_secs=600,
                                               checkpoint_dir=args.model_path,
                                               scaffold=scaffold) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():

                    ppl, loss, _, step = sess.run([train_ppl, train_loss, train_op,
                                                   global_step])  # ,options=run_options, run_metadata=run_metadata)
                    if step % args.steps_to_validate == 0:
                        va_cost = []
                        va_ppl = []
                        for xm, mm in iter_data((vaX, vaM), n_batch=model.n_batch_train, truncate=False,
                                                verbose=True):
                            res, ppl = sess.run([valid_loss, valid_ppl],
                                                {X: xm, M: mm})
                            va_cost.append(np.sum(res))
                            va_ppl.append(np.sum(ppl))
                        ps = sess.run(model.params)
                        tf.logging.info('saving LM params into ' + args.lm_path)
                        joblib.dump(ps, args.lm_path + '/model_lm.params', protocol=2)
                        va_cost = np.average(va_cost)
                        va_ppl = np.average(va_ppl)
                        tf.logging.info('=========n_steps:\t%d valid_cost:\t%.3f valid ppl:\t%.3f==========' % (
                        step, va_cost, va_ppl))

            except tf.errors.OutOfRangeError:
                print('Epochs Complete!')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    if args.preprocess:
        preprocess(args)
        model = LM_transformer_pretrain(args)
        model.encode_to_tfrecords(tfrecord_filename=args.tfrecord_filename,origin_filename=args.train_data_path)
        print('preprocess finished')
    else:
        model = LM_transformer_pretrain(args)
        ccc_train(model,args)

