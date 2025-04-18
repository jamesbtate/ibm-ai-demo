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

# from https://github.com/IBM/ibmz-accelerated-serving-for-tensorflow/blob/main/samples/fashion-mnist/fashion_mnist_rest.py

import json
import time
import tensorflow as tf

from six.moves import urllib

def request_inference(input_data):
    print(time.time(), "in function")
    req = {"signature_name": "serving_default",
           "inputs": input_data}
    data = json.dumps(req).encode('utf-8')
    url = 'http://localhost:8501/v1/models/fashion_mnist:predict'

    start_time = time.time()
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, data=data))
        print(time.time(), "Made request")
        resp_data = resp.read()
        print(time.time(), "got response")
        resp.close()
    except Exception as e:  # pylint: disable=broad-except
        print('Request failed. Error: {}'.format(e))
        raise e
    stop_time = time.time()
    elapsed = stop_time - start_time
    print(f"End-to-end time: {round(elapsed,2)}s")

    # Process output.
    output = json.loads(resp_data.decode())['outputs']
    y_pred = tf.convert_to_tensor(output, dtype=tf.float32)
    print(time.time(), "converted output")
    return y_pred, elapsed

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_test = X_test.astype('float32') / 255
print(f"Loaded X_test with {len(X_test)} items")

# Reshape from [N, 28, 28] to expected input shape [N, 28, 28, 1]
X_test = tf.expand_dims(X_test, axis=-1)

print(f"Doing inference on all 10k items...")
y_pred, elapsed = request_inference(X_test.numpy().tolist())
print(f"y_pred shape: {type(y_pred)} [0]{type(y_pred[0])} {y_pred.shape}")

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_test, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
print('Test accuracy:', accuracy.numpy())
