# MIT License, see LICENSE
# Copyright (c) 2019 Paperspace Inc.
# Author: Michal Kulaczkowski

from __future__ import print_function
import base64
import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf
from tensorflow.python.training import server_lib

from utils import train_dataset, test_dataset, ping


def parse_args():
    """Parse arguments"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='''Train a convolution neural network with MNIST dataset.
                            For distributed mode, the script will use few environment variables as defaults:
                            JOB_NAME, TASK_INDEX, PS_HOSTS, and WORKER_HOSTS. These environment variables will be
                            available on distributed Tensorflow jobs on Paperspace platform by default.
                            If running this locally, you will need to set these environment variables
                            or pass them in as arguments (i.e. python mnist.py --job_name worker --task_index 0
                            --worker_hosts "localhost:2222,localhost:2223" --ps_hosts "localhost:2224").
                            If these are not set, the script will run in non-distributed (single instance) mode.''')

    # Configuration for distributed task
    parser.add_argument('--job_name', type=str, default=os.environ.get('TYPE', None), choices=['worker', 'ps', 'master'],
                        help='Task type for the node in the distributed cluster. Worker-0 will be set as master.')
    parser.add_argument('--task_index', type=int, default=os.environ.get('INDEX', 0),
                        help='Worker task index, should be >= 0. task_index=0 is the chief worker.')
    parser.add_argument('--ps_hosts', type=str, default=os.environ.get('PS_HOSTS'),
                        help='Comma-separated list of hostname:port pairs.')
    parser.add_argument('--worker_hosts', type=str, default=os.environ.get('WORKER_HOSTS'),
                        help='Comma-separated list of hostname:port pairs.')
    parser.add_argument('--master', type=str, default=os.environ.get('MASTER'),
                        help='Comma-separated list of hostname:port pairs.')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Number of steps to run trainer.')
    # Experiment related parameters
    parser.add_argument('--local_data_root', type=str, default=os.path.abspath(os.getenv('PS_HOME', os.getcwd()) + '/data'),
                        help='Path to dataset. This path will be /data on Paperspace.')
    parser.add_argument('--local_log_root', type=str, default=os.path.abspath(os.getenv('PS_HOME', os.getcwd()) + '/logs'),
                        help='Path to store logs and checkpoints. This path will be /artifacts on Paperspace.')
    parser.add_argument('--data_subpath', type=str, default='',
                        help='Which sub-directory the data will sit inside local_data_root (locally) ' +
                             'or /data/ (on Paperspace).')

    # CNN model params
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Size of the CNN kernels to use.')
    parser.add_argument('--hidden_units', type=str, default='32,64',
                        help='Comma-separated list of integers. Number of hidden units to use in CNN model.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate used in Adam optimizer.')
    parser.add_argument('--learning_decay', type=float, default=0.0001,
                        help='Exponential decay rate of the learning rate per step.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate used after each convolutional layer.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size to use during training and evaluation.')

    # Training params
    parser.add_argument('--verbosity', type=str, default='DEBUG', choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'],
                        help='TF logging level. To see intermediate results printed, set this to INFO or DEBUG.')
    parser.add_argument('--fashion', action='store_true',
                        help='Download and use fashion MNIST data instead of the default handwritten digit MNIST.')
    parser.add_argument('--parallel_batches', type=int, default=2,
                        help='Number of parallel batches to prepare in data pipeline.')
    parser.add_argument('--max_ckpts', type=int, default=2,
                        help='Maximum number of checkpoints to keep.')
    parser.add_argument('--ckpt_steps', type=int, default=100,
                        help='How frequently to save a model checkpoint.')
    parser.add_argument('--save_summary_steps', type=int, default=10,
                        help='How frequently to save TensorBoard summaries.')
    parser.add_argument('--log_step_count_steps', type=int, default=10,
                        help='How frequently to log loss & global steps/s.')
    parser.add_argument('--eval_secs', type=int, default=60,
                        help='How frequently to run evaluation step.')

    # Parse args
    opts = parser.parse_args()

    tf.logging.set_verbosity(opts.verbosity)

    opts.data_dir = os.path.abspath(os.environ.get('PS_JOBSPACE', os.getcwd()) + '/data')
    opts.log_dir = os.path.abspath(os.environ.get('PS_JOBSPACE', os.getcwd()) + '/logs')

    if not os.path.exists(opts.data_dir):
        os.mkdir(opts.data_dir)
    if not os.path.exists(opts.log_dir):
        os.mkdir(opts.log_dir)

    opts.hidden_units = [int(n) for n in opts.hidden_units.split(',')]

    tf.logging.info('************STORAGE CHECK*******************')
    # print storage options:
    files = os.listdir(os.environ.get('PS_JOBSPACE'))

    tf.logging.info('DATA in PS_JOBSPACE')
    for name in files:
        tf.logging.info(name)

    # print storage options:
    files = os.listdir(os.environ.get('PS_HOME'))
    tf.logging.info('DATA in PS_HOME')
    for name in files:
        tf.logging.info(name)

    return opts


def get_tf_config():
    tf_config = json.loads(os.environ.get('TF_CONFIG'))
    if not tf_config:
        return
    return tf_config


def check_ps_nodes(config):
    tf_config = get_tf_config()
    if not tf_config:
        return
    assert config.task_id == tf_config['task']['index']
    assert config.task_type == 'ps'
    assert not config.is_chief


def check_master_nodes(config):
    tf_config = get_tf_config()
    if not tf_config:
        return
    assert config.task_id == tf_config['task']['index']
    assert config.task_type == 'master'
    assert config.is_chief


def check_worker_nodes(config):
    tf_config = get_tf_config()
    if not tf_config:
        return
    assert config.task_id == tf_config['task']['index']
    assert config.task_type == 'worker'
    assert not config.is_chief


def check_clusterspec(config):
    tf_config = get_tf_config()
    if not tf_config:
        return
    assert config.num_ps_replicas == len(tf_config['cluster']['ps'])
    assert config.num_worker_replicas == len(tf_config['cluster']['worker']) + len(tf_config['cluster']['master'])
    assert config.cluster_spec == server_lib.ClusterSpec(tf_config['cluster'])


def get_paperspace_tf_config(args):
    tf_config = os.environ.get('TF_CONFIG')
    if not tf_config:
        return
    paperspace_tf_config = json.loads(base64.urlsafe_b64decode(tf_config).decode('utf-8'))
    
    tf.logging.debug(str(paperspace_tf_config))
    return paperspace_tf_config


def get_input_fn(opts, is_train=True):
    """Returns input_fn.  is_train=True shuffles and repeats data indefinitely"""

    def input_fn():
        with tf.device('/cpu:0'):
            if is_train:
                dataset = train_dataset(opts.data_dir, fashion=opts.fashion)
                dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=5 * opts.batch_size, count=None))
            else:
                dataset = test_dataset(opts.data_dir, fashion=opts.fashion)
            dataset = dataset.batch(batch_size=opts.batch_size)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()

    return input_fn


def cnn_net(input_tensor, opts):
    """Return logits output from CNN net"""
    temp = tf.reshape(input_tensor, shape=(-1, 28, 28, 1), name='input_image')
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.layers.conv2d(temp, filters=n_units, kernel_size=opts.kernel_size, strides=(2, 2),
                                activation=tf.nn.relu, name='cnn' + str(i))
        temp = tf.layers.dropout(temp, rate=opts.dropout)
    temp = tf.reduce_mean(temp, axis=(2, 3), keepdims=False, name='average')
    return tf.layers.dense(temp, 10)


def get_model_fn(opts):
    """Return model fn to be used for Estimator class"""

    def model_fn(features, labels, mode):
        """Returns EstimatorSpec for different mode (train/eval/predict)"""
        logits = cnn_net(features, opts)
        pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions={'logits': logits, 'pred': pred})

        cent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
        loss = tf.reduce_mean(cent, name='loss')

        metrics = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=pred, name='accuracy')}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        optimizer = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_fn


def main(opts):
    """Main"""
    # Create an estimator
    # distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
    #     num_gpus_per_worker=1)

    config = tf.estimator.RunConfig(
        model_dir=opts.log_dir,
        save_summary_steps=opts.save_summary_steps,
        save_checkpoints_steps=opts.ckpt_steps,
        keep_checkpoint_max=opts.max_ckpts,
        # train_distribute=distribution,
        log_step_count_steps=opts.log_step_count_steps)

    # CHECK CONFIG
    check_clusterspec(config)
    type = os.environ.get('TYPE')
    if type in ('chief','master'):
        check_master_nodes(config)
    elif type == 'worker':
        check_worker_nodes(config)
    elif type == 'ps':
        check_ps_nodes(config)

    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(opts),
        config=config)

    # Create input fn
    # We do not provide evaluation data, so we'll just use training data for both train & evaluation.
    train_input_fn = get_input_fn(opts, is_train=True)
    eval_input_fn = get_input_fn(opts, is_train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=opts.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=opts.eval_secs)

    # Train and evaluate!
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    args = parse_args()

    tf.logging.debug('=' * 20 + ' Environment Variables ' + '=' * 20)
    for k, v in os.environ.items():
        tf.logging.debug('{}: {}'.format(k, v))

    tf.logging.debug('=' * 20 + ' Arguments ' + '=' * 20)
    for k, v in sorted(args.__dict__.items()):
        if v is not None:
            tf.logging.debug('{}: {}'.format(k, v))

    tf_config = get_paperspace_tf_config(args)
    if tf_config:
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

    tf.logging.info('=' * 20 + ' Train starting ' + '=' * 20)
    main(args)
