---
name: run-write-model-report
description: Your goal is to execute the script for running a model that is given to you and write a report of the model's status - does it work and if not, why not.
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, Task
---

The goal is to run a particular model script given by the user and write a report of the configuration used, hardware used and the model status. If the script didnt include dumping the log into a particular file (with something like ` &> log.txt`), then add that to the script, as logs can be huge. Don't put a timeout on script execution, as they can last long. The report should contain the following information:

1. Write the configuration of the model used
- Which model are we testing - specific name and the variant (e.g. Llama-3.1-8B-Instruct-Turbo-4096, or StabilityAI/stable-diffusion-xl-base-1.0). Also make sure to see which parts of the model are "on device". Meaning only check which parts of the model are being put on the tenstorrent hardware. For example, Stable Diffusion XL has Unet, VAE, Text Encoders which can all be on cpu or tt hardware, but 
look for the lines in the inference script like model.to(device) where device is the tenstorrent hardware. This also might depend on the particular command line arguments used to run the model. Also look for which optimization level is used in torch_xla.set_custom_compile_options(
        {"optimization_level": opt}
    )

2. Write about the hardware used 
Use tt-smi -ls to get the information about the hardware used. If tt-smi is not installed, use pip install tt-smi to install it. Always activate the enviroment first using source venv/activate.
Get information about the Device Series (e.g. p100a, n150, etc.)

3. Model status
This is the information about the model execution status - whether it successfully ran or not. If you found some errors or failures in the log, you can copy the exact error messages. if not, put a "Success".

The report should be written in a yaml format. The format is like:

model_name: Llama-3.1-8B-Instruct-Turbo-4096
weights_dtype: bfloat16
optimization_level: 1
model_parts_on_device: Unet, Text Encoders
device: p100a
model_success: false
error_messages: Error: OperationValidationAndFallback: Operation ttnn.conv2d failed validation

Please run the model and write a report in this format.