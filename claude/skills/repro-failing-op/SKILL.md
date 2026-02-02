---
name: repro-failing-op
description: Get a repro of failing op in the model. Use when debugging model failures and realizing that the error is caused by a concrete op in the model that is failing (any type of op error).
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, Task
---

### Overview
The goal is to try to make a minimal reproducible example of the op that is failing in the model. The error log is most likely created by running a python script that loads the model and performs inference. Your goal is to create a new python file that can result in the same error, but only loading one (or a few) operations. These operations are either torch or jax ops. 

### Example:

If the model X is failing on torch Groupnorm operation, a viable reproduction test case might be

```
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import UNet2DConditionModel

def run_groupnorm_test():
    xr.set_device_type("TT")
    whole_model = UNet2DConditionModel.from_pretrained(<model_id>)
    model = whole_model.path.to.group_norm # you can find this path from locations in the .log file
    # alternatively, use model = nn.GroupNorm(<same params>) if you cannot reproduce by taking module of original model
    model = model.to(torch.bfloat16)
    model = model.eval()
    
    input = torch.randn(1, 512, 16, 16, dtype=torch.bfloat16) + 10.0
    
    # Get torch reference output (on CPU)
    print(f"Computing torch reference output...")
    with torch.no_grad():
        torch_output = model(input)
    
    # Compile with TT backend
    print(f"Compiling model with backend='tt'...")
    model = torch.compile(model, backend='tt')
    
    # Move to device and run
    device = xm.xla_device()
    input_device = input.to(device)
    model = model.to(device)
    
    print(f"Running on TT device...")
    with torch.no_grad():
        tt_output = model(input_device)
    
    tt_output_cpu = tt_output.cpu()
    print(tt_output_cpu)


if __name__ == "__main__":
    run_groupnorm_test()
```

### Running the script 
source venv/activate
// these are useful flags for debugging
TTXLA_LOGGER_LEVEL=DEBUG XLA_HLO_DEBUG=1 TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG python3 path/to/script.py &> minimal.log

### Debugging tips 
If you are not getting the reproduction successfully, look at the original model test/example and your atempt to reproduce. Maybe there are additional changes. 
E.g. a common think can be to turn set optimization level as `torch_xla.set_custom_compile_options(
        {"optimization_level": [0|1|2]}
    )`


### Success criteria
If the output that you get from the op test case shows the same error as the test for the model, that means you successfully made a repro.

### Required outputs
1. Python script that successfully reproduces the error
2. minimal.mlir file that contains the ttir representation of the op/graph that can reproduce the error. Here is the example of that format 
```#loc1 = loc("p0.1")
#loc2 = loc("p1.3")
module @SyncTensorsGraph.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<320x960x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "fn_weight"} loc("p0.1"), %arg1: tensor<2x960x128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p1.3")) -> (tensor<2x320x128x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = "ttir.convolution"(%arg1, %arg0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<2x960x128x128xbf16>, tensor<320x960x3x3xbf16>) -> tensor<2x320x128x128xbf16> loc(#loc3)
        return %0 : tensor<2x320x128x128xbf16> loc(#loc)  
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("convolution.5")```