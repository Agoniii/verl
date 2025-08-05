## vllm backend

### VLLM 0.9.1 修改:
```
diff --git a/model_executor/layers/quantization/fp8.py b/model_executor/layers/quantization/fp8.py
index dc9c5cb..75a4aed 100644
--- a/model_executor/layers/quantization/fp8.py
+++ b/model_executor/layers/quantization/fp8.py
@@ -63,7 +63,7 @@ class Fp8Config(QuantizationConfig):
         weight_block_size: Optional[list[int]] = None,
     ) -> None:
         super().__init__()
-
+        is_checkpoint_fp8_serialized = True
         self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

diff --git a/model_executor/layers/quantization/utils/w8a8_utils.py b/model_executor/layers/quantization/utils/w8a8_utils.py
index adc67aa..361316b 100644
--- a/model_executor/layers/quantization/utils/w8a8_utils.py
+++ b/model_executor/layers/quantization/utils/w8a8_utils.py
@@ -34,6 +34,7 @@ def sparse_cutlass_supported() -> bool:


 def cutlass_fp8_supported() -> bool:
+    return False
     if not current_platform.is_cuda():
         return False
```
### block quant

1. set use_block_quant=True
`verl/trainer/config/ppo_megatron_trainer.yaml`
```
  rollout:
    use_block_quant_rollout: True
```

### per tensor

1. set use_block_quant=False
`verl/trainer/config/ppo_megatron_trainer.yaml`
```
  rollout:
    use_block_quant_rollout: False
```

## sglang backend

### sglang0.4.8 修改  
refer to https://github.com/Agoniii/sglang/commit/81ecca0b4aac5a662d9bc6d7b72edfe0d138a713

### block quant 

`verl/workers/sharding_manager/fp8_util.py`

```
def quant_weights_sglang(weights):
    use_block_quant = True
```

### per tensor(still has error)

1. `verl/workers/sharding_manager/fp8_util.py`

```
def quant_weights_sglang(weights):
    use_block_quant = True
```

2. `/usr/local/lib/python3.10/dist-packages/sglang/srt/configs/model_config.py`
```
diff --git a/srt/configs/model_config.py b/srt/configs/model_config.py
index 6ddd248..98c65b6 100644
--- a/srt/configs/model_config.py
+++ b/srt/configs/model_config.py
@@ -81,6 +81,14 @@ class ModelConfig:
             model_override_args=self.model_override_args,
             **kwargs,
         )
+        # FP8_BLOCK_QUANT_KWARGS = {
+        #     "activation_scheme": "dynamic",
+        #     "fmt": "e4m3",
+        #     "quant_method": "fp8",
+        #     "weight_block_size": [128, 128],
+        # }
+        # fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
+        # setattr(self.hf_config, "quantization_config", fp8_block_quant_kwargs)
```
error log:
```
(WorkerDict pid=1650695)   File "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/linear.py", line 445, in forward
(WorkerDict pid=1650695)     output_parallel = self.quant_method.apply(self, input_, bias)
(WorkerDict pid=1650695)   File "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/quantization/fp8.py", line 519, in apply
(WorkerDict pid=1650695)     weight_scale, weight = requantize_with_max_scale(
(WorkerDict pid=1650695)   File "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/quantization/utils.py", line 111, in requantize_with_max_scale
(WorkerDict pid=1650695)     if unfused_module_in_checkpoint:
(WorkerDict pid=1650695) RuntimeError: CUDA error: operation not permitted when stream is capturing
(WorkerDict pid=1650695) CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(WorkerDict pid=1650695) For debugging consider passing CUDA_LAUNCH_BLOCKING=1
(WorkerDict pid=1650695) Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```