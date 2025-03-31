# Copyright 2023 IBM Corp.
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

"""
Sample Tensorflow application with Fashion MNIST model trained by
`fashion_mnist_training.py`.

Modified to use arguments for a little configuration.
"""
import tensorflow as tf
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=64,
                    help='Number of images in the batch. Default is 64')
parser.add_argument('-t', '--threads', type=int, default=0,
                    help='Tensorflow op parallelism limit. Default is 0 (no limit)')
parser.add_argument('-r', '--runs', type=int, default=1,
                    help='Number of times to do the inference. Default is 1')
args = parser.parse_args()

if args.threads > 0:
    print(f"Running with {args.threads} threads.")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)

(X_train, y_train), (X_test, y_test) = \
    tf.keras.datasets.fashion_mnist.load_data()

X_test = X_test.astype('float32') / 255

model = tf.keras.models.load_model('model.keras')
print(model.summary())

for i in range(args.runs):
    start_time = time.time()
    y_pred = model.predict(X_test, batch_size=args.batch_size)
    stop_time = time.time()

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_test, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
    print(f'Runtime: {round(stop_time-start_time, 3)}s  Test accuracy: {round(float(accuracy.numpy()*100), 2)}%')
