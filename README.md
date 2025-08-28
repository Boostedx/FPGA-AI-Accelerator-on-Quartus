⚡ FPGA-AI-Accelerator-on-Quartus

Accelerating AI inference with custom FPGA designs on Intel Quartus Prime

📖 Overview

This repository documents my work on building a lightweight AI accelerator using Intel Quartus Prime. The project focuses on implementing a Convolutional Neural Network (CNN) model on FPGA hardware, optimized for:

✅ Low-latency inference

✅ Efficient memory usage on resource-constrained devices

✅ Custom datapath design for matrix multiplication and convolution

✅ Scalable OpenCL/HLS workflows for rapid experimentation

The accelerator has been tested on Intel Cyclone V (DE1-SoC), targeting edge AI applications where real-time performance and energy efficiency are critical.

🚀 Features

End-to-end flow: from Python-trained weights → HLS4ML/OpenCL kernels → Quartus synthesis.

Support for quantized models (8-bit / 16-bit).

Memory-aware design with pruning support.

Benchmarked on MNIST classification, achieving >97% accuracy on FPGA.

🛠️ Tech Stack

Intel Quartus Prime (18.1+)

OpenCL SDK / HLS Compiler

HLS4ML for model translation

Cyclone V FPGA (DE1-SoC) testbed

Python (TensorFlow / PyTorch) for model training

