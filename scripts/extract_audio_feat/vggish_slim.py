# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines the 'VGGish' model used to generate AudioSet embedding features.

The public AudioSet release (https://research.google.com/audioset/download.html)
includes 128-D features extracted from the embedding layer of a VGG-like model
that was trained on a large Google-internal YouTube dataset. Here we provide
a TF-Slim definition of the same model, without any dependences on libraries
internal to Google. We call it 'VGGish'.

Note that we only define the model up to the embedding layer, which is the
penultimate layer before the final classifier layer. We also provide various
hyperparameter values (in vggish_params.py) that were used to train this model
internally.

For comparison, here is TF-Slim's VGG definition:
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
"""
import os # 新增
import urllib.request # 新增
import tensorflow as tf
import vggish_params as params

# slim = tf.contrib.slim
import tf_slim as slim


def define_vggish_slim(training=False):
    """Defines the VGGish TensorFlow model.

    All ops are created in the current default graph, under the scope 'vggish/'.

    The input is a placeholder named 'vggish/input_features' of type float32 and
    shape [batch_size, num_frames, num_bands] where batch_size is variable and
    num_frames and num_bands are constants, and [num_frames, num_bands] represents
    a log-mel-scale spectrogram patch covering num_bands frequency bands and
    num_frames time frames (where each frame step is usually 10ms). This is
    produced by computing the stabilized log(mel-spectrogram + params.LOG_OFFSET).
    The output is an op named 'vggish/embedding' which produces the activations of
    a 128-D embedding layer, which is usually the penultimate layer when used as
    part of a full model with a final classifier layer.

    Args:
      training: If true, all parameters are marked trainable.

    Returns:
      The op 'vggish/embeddings'.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.compat.v1.truncated_normal_initializer(  # 修改点
                            stddev=params.INIT_STDDEV),
                        biases_initializer=tf.compat.v1.zeros_initializer(),
                        activation_fn=tf.nn.relu,
                        trainable=training), \
            slim.arg_scope([slim.conv2d],
                           kernel_size=[3, 3], stride=1, padding='SAME'), \
            slim.arg_scope([slim.max_pool2d],
                           kernel_size=[2, 2], stride=2, padding='SAME'), \
            tf.compat.v1.variable_scope('vggish'):
        features = tf.compat.v1.placeholder(
            tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
            name='input_features')
        net = tf.reshape(features, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])

        net = slim.conv2d(net, 64, scope='conv1')
        net = slim.max_pool2d(net, scope='pool1')
        net = slim.conv2d(net, 128, scope='conv2')
        net = slim.max_pool2d(net, scope='pool2')
        net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
        net = slim.max_pool2d(net, scope='pool3')
        net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
        net = slim.max_pool2d(net, scope='pool4')

        net = slim.flatten(net)
        net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
        net = slim.fully_connected(net, params.EMBEDDING_SIZE, scope='fc2')
        return tf.compat.v1.identity(net, name='embedding')  # 修改点


def load_vggish_slim_checkpoint(session, checkpoint_path):
    """Loads a pre-trained VGGish-compatible checkpoint.

    This function can be used as an initialization function (referred to as
    init_fn in TensorFlow documentation) which is called in a Session after
    initializating all variables. When used as an init_fn, this will load
    a pre-trained checkpoint that is compatible with the VGGish model
    definition. Only variables defined by VGGish will be loaded.

    Args:
      session: an active TensorFlow session.
      checkpoint_path: path to a file containing a checkpoint that is
        compatible with the VGGish model definition.
    """
    # 新增：检查权重文件是否存在，如果不存在则下载
    vggish_model_url = 'https://storage.googleapis.com/audioset/vggish_model.ckpt'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} not found. Downloading from {vggish_model_url}...")
        try:
            # 确保 checkpoint_path 所在的目录存在
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
            urllib.request.urlretrieve(vggish_model_url, checkpoint_path)
            print(f"Successfully downloaded {checkpoint_path}")
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            raise # 如果下载失败，则抛出异常

    # Get the list of names of all VGGish variables that exist in
    # the checkpoint (i.e., all inference-mode VGGish variables).
    with tf.Graph().as_default():
        define_vggish_slim(training=False)
        vggish_var_names = [v.name for v in tf.compat.v1.global_variables()]

    # Get the list of all currently existing variables that match
    # the list of variable names we just computed.
    vggish_vars = [v for v in tf.compat.v1.global_variables() if v.name in vggish_var_names]

    # Use a Saver to restore just the variables selected above.
    saver = tf.compat.v1.train.Saver(vggish_vars, name='vggish_load_pretrained')
    saver.restore(session, checkpoint_path)