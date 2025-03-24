This repo has ML test code for pytorch/tensorflow. It was initially
based on example code provisded by IBM in these repos:
https://github.com/IBM/ibmz-accelerated-for-tensorflow/tree/main
https://github.com/IBM/ibmz-accelerated-for-pytorch

To test the fashion MNIST example using Tensorflow:
```
# Create a virtual env and install tensorflow
python -m venv venv
source venv/bin/activate
pip install tensorflow

# train the model (saves to file in CWD)
python fashion_mnist_training.py

# test inference performance
# this will automatically use IBM z16 NNPA if available
python fashion_mnist.py
# to force it not to use the NNPA, use this environment variable:
export NNPA_DEVICES=0
# not sure what to do to make this code use CUDA
```

To test the cat/dog image classification (using PyTorch):
This code requires the Python `tkinter` library. On Windows, this is typically
installed as part of Python. On Linux, it is typically installed as a separate system
package with apt/yum. Of course as it is a GUI you must run it locally
on your workstation or use X11 forwarding to get the output from a remote
machine.
```
# Create a virtual env and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# train the model - the model is garbage so stop after 1 epoch
python catdog_training.py --resize 256 --model-path model_256.pt --epochs 1

# concurrently test the performance of PCU inference and NNPA inference
python comparison_ui.py
```
This code has a few flags available to manipulation the configuration.
If the NNPA is not available, both "sides" of the comparison will use
the CPU - obviously that is a bizarre performance test. TO disable the
NNPA "side" use the --cpu-only flag.

This code could be modified to use a `cuda` device instead of NNPA/CPU.
One way to do that would be to modify the function `run_nnpa_process(...)`. In the call
to `catdog_training.test_setup(...)` change the device from `'nnpa'` to `'cuda'`.
Then you can do a comparison of CPU vs CUDA concurrently or use the
`--cpu-only` or `--nnpa-only` flags to do just one engine or the other.
