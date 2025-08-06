import torch, os, ray
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoModel
try:
    from vllm.model_executor.layers.linear import LinearBase
    from vllm._custom_ops import scaled_fp8_quant
    print("Vllm FP8 quantization available")
except Exception:
    try:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
        print("SGLang FP8 quantization available")
    except Exception:
        print("FP8 quantization not available")

#from vllm.triton_utils import tl, triton
from unittest.mock import patch

# from vllm.v1.engine.core import EngineCoreProc
# from vllm.v1.utils import CoreEngineProcManager
try:
    from vllm import _custom_ops as ops
except Exception:
    print("")

FP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


@dataclass()
class FP8State:
    # A cache of fp8 parameter names, we can check this cache to see if a
    # param name corresponds to a fp8 weight
    seen_params: set = field(default_factory=lambda: set())
    fp8_param_names: set = field(default_factory=lambda: set())
    vllm_patches: list = field(default_factory=lambda: [])

fp8_state: FP8State = FP8State()


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if hasattr(vllm_config, "quant_config") and isinstance(
        vllm_config.quant_config, Fp8Config
    ):
        # assert vllm_config.quant_config.weight_block_size is not None, (
        #     "Only block scaling is currently supported in NeMo-RL!"
        # )
        return True

    return False


def _get_module_from_param_name(model, name: str):
    # Split the name into parts (e.g., 'layers', '0', 'self_attn', 'q_proj', 'weight')
    # The module path is all but the last part (the parameter's own name)
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    # Replace with the fused model name
    packed_modules_mapping = model.packed_modules_mapping
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path[-1] in reversed_mapping.keys():
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        # Traverse the model hierarchy
        for part in module_path:
            if isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not find module for parameter '{name}'. Error: {e}")
    return current_module


def _is_fp8_weight(name, model):
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = _get_module_from_param_name(model, name)
            # We currently only quantize linear layers
            if (
                isinstance(module, LinearBase)
                and module.weight.dtype == torch.float8_e4m3fn
            ):
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names

def _is_fp8_weight_sglang(name, tensor):
    is_linear_weight = 'q_proj.weight' in name or 'k_proj.weight' in name or \
        'v_proj.weight' in name or 'o_proj.weight' in name or \
        'gate_proj.weight' in name or 'up_proj.weight' in name or \
        'down_proj.weight' in name
    is_not_fp8 = tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]
    return is_linear_weight and is_not_fp8

def kitchen_block_scale(
    data_hp,
    weight_block_size,
):
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    assert data_hp.shape[1] % block_size1 == 0, (
        f"data_hp.shape[1] {data_hp.shape[1]}  must be a multiple of block_size1: {block_size1}."
    )
    assert data_hp.shape[0] % block_size0 == 0, (
        f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."
    )

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    original_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    assert block_size1 == block_size0
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data_hp = data_hp.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)
    # Calculate descale factor
    descale = max_abs / max_dtype

    # global global_fp8_config
    # if global_fp8_config.use_weight_pow2_scale:
    #     exponent = torch.ceil(torch.log2(descale))
    #     # Post process exponent to be in range of -127 to 127 and to be E8M0 biased
    #     exponent = torch.clamp(exponent, min=-127, max=127) + 127
    #     # Convert to uint8 container
    #     exponent = exponent.to(torch.uint8)
    #     # Calculate descale_fp to apply to data_hp
    #     scale_fp = torch.where(
    #         # If exponent is 0, descale_fp is 1.0 rather than 2^127
    #         exponent == 0,
    #         1.0,
    #         torch.exp2(127 - exponent.to(torch.float32)),
    #     )
    #     descale_fp = torch.reciprocal(scale_fp)
    # else:
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    # preserve the behavior for 0 amax case
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)

    descale_fp = torch.reciprocal(scale_fp)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(original_shape)
    )

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp

def quant_weights_sglang(weights):
    use_block_quant = False
    weights_quantized = []
    qkv_weights = []
    gate_up_weights = []

    for k, v in weights:
        if not _is_fp8_weight_sglang(k, v):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into fp8 and its scale factor
        # print(f"xueh {quant_config.weight_block_size=}")
        
        if use_block_quant:
            print("Using blockwise quantization")
            param_lp, param_scale = kitchen_block_scale(
                v.to(torch.float),
                weight_block_size=FP8_BLOCK_QUANT_KWARGS["weight_block_size"],
            )
            param_scale = param_scale.squeeze(-1)
            # print(f"xueh quant {k} param: {param_lp.shape}, {param_lp[:10][:10]}")
            # print(f"xueh quant {k} scale: {param_scale.shape}, {param_scale}")
            weights_quantized.append([k, param_lp])
            weights_quantized.append([k + "_scale_inv", param_scale])
        else:
            print("Using Per tensor quantization")
            original_shape = v.shape
            # Apply FP8 quantization using SGLang's online quantization
            # force to use per-tensor quant for weights
            quantized_tensor, scale = scaled_fp8_quant(v)
            # Reshape back to original shape
            quantized_tensor = quantized_tensor.view(original_shape)

            # TODO: need to check why only o_proj and down_proj need to do transpose
            # if 'o_proj.weight' in k or 'down_proj.weight' in k:
            #     quantized_tensor = quantized_tensor.t()

            scale_k = k.replace(".weight", ".weight_scale")
            scale = scale.view(1)
            weights_quantized.extend([(k, quantized_tensor), (scale_k, scale)])
            # if "layers.0" in k:
            #     print(f"Applied FP8 quantization to tensor {k} {quantized_tensor.shape} {quantized_tensor[:10][:10]}")
            #     print(f"Applied FP8 quantization to tensor {k} {scale.shape} {scale}")
    return weights_quantized


def quant_weights(weights, model, quant_config):
    weights_quantized = []
    qkv_weights = []
    gate_up_weights = []
    for k, v in weights:
        if not _is_fp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into fp8 and its scale factor
        # print(f"xueh {quant_config.weight_block_size=}")
        if quant_config.weight_block_size is not None:
            print("Using blockwise quantization")
            param_lp, param_scale = kitchen_block_scale(
                v.to(torch.float),
                weight_block_size=quant_config.weight_block_size,
            )
            param_scale = param_scale.squeeze(-1)
            # print(f"xueh quant {k} param: {param_lp.shape}, {param_lp[:10][:10]}")
            # print(f"xueh quant {k} scale: {param_scale.shape}, {param_scale}")
            weights_quantized.append([k, param_lp])
            weights_quantized.append([k + "_scale_inv", param_scale])

        else:
            print("Using Per tensor quantization")
            # if 'q_proj.weight' in k:
            #     layer_id = int(k.split(".")[2])
            #     k_weight = weights[k.replace(".q_proj", ".k_proj")]
            #     v_weight = weights[k.replace(".q_proj", ".v_proj")]
            #     qkv_weights[layer_id].append((k.replace(".q_proj", ".qkv_proj"), torch.cat([v, k_weight, v_weight])))
            # elif 'gate_proj.weight' in k:
            #     layer_id = int(k.split(".")[2])
            #     up_weight = weights[k.replace(".gate_proj", ".up_proj")]
            #     gate_up_weights[layer_id].append((k.replace(".gate_proj", ".gate_up_proj"), torch.cat([v, up_weight])))
            # elif 'k_proj.weight' in k or 'v_proj.weight' in k or 'up_proj.weight' in k:
            #     continue

            original_shape = v.shape
            # Apply FP8 quantization using SGLang's online quantization
            # force to use per-tensor quant for weights
            quantized_tensor, scale = scaled_fp8_quant(v)
            # Reshape back to original shape
            quantized_tensor = quantized_tensor.view(original_shape)

            # TODO: need to check why only o_proj and down_proj need to do transpose
            # if 'o_proj.weight' in k or 'down_proj.weight' in k:
            #     quantized_tensor = quantized_tensor.t()

            scale_k = k.replace(".weight", ".weight_scale")
            scale = scale.view(1)
            weights_quantized.extend([(k, quantized_tensor), (scale_k, scale)])
            # if "layers.0" in k:
            #     print(f"Applied FP8 quantization to tensor {k} {quantized_tensor.shape} {quantized_tensor[:10][:10]}")
            #     print(f"Applied FP8 quantization to tensor {k} {scale.shape} {scale}")

    # if len(qkv_weights) > 0 and len(gate_up_weights) > 0:
    #     for k, v in qkv_weights:
    #         original_shape = v.shape
    #         quantized_tensor, scale = scaled_fp8_quant(v)
    #         quantized_tensor = quantized_tensor.view(original_shape)
    #         scale_k = k.replace(".weight", ".weight_scale")
    #         scale = scale.view(1)
    #         weights_quantized.extend([(k, quantized_tensor), (scale_k, scale)])
    #         if "layers.0" in k:
    #             print(f"Applied FP8 quantization to tensor {k} {quantized_tensor.shape} {quantized_tensor[:10][:10]}")
    #             print(f"Applied FP8 quantization to tensor {k} {scale.shape} {scale}")

    return weights_quantized


def load_quanted_weights(weights, model_runner):
    #weights_quantized = []
    model = model_runner.model
    quant_config = model_runner.vllm_config.quant_config

    weights_quantized = quant_weights(weights, model, quant_config)

    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weightloader
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm
    loaded_params = model.load_weights(weights_quantized)
    # Undo the type change above to the original type
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.__class__ = param.orig_type
    return loaded_params


def process_weights_after_loading(self, layer) -> None:
    print(f"xueh patch process_weights_after_loading {self.block_quant=}, {self.quant_config.is_checkpoint_fp8_serialized=}")
    try:
        from vllm.model_executor.parameter import (
            BlockQuantScaleParameter,
            ModelWeightParameter,
            PerTensorScaleParameter
        )
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale
    except Exception:
        try:
            from sglang.srt.layers.parameter import (
                BlockQuantScaleParameter,
                ModelWeightParameter,
                PerTensorScaleParameter
            )
            from sglang.srt.layers.quantization.utils import requantize_with_max_scale
        except Exception:
            print("error")
    from torch.nn import Parameter

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr
            for attr in custom_param_dir
            if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
        return param

    if self.block_quant:
        assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
        assert self.quant_config.activation_scheme == "dynamic"
        weight = layer.weight.data
        weight_scale_inv = layer.weight_scale_inv.data
        weight = self._maybe_pad_weight(weight)

        layer.weight = _create_param_from_subclass_attributes(
            ModelWeightParameter(
                data=weight,
                output_dim=0,
                input_dim=1,
                weight_loader=layer.weight.weight_loader,
            )
        )
        layer.weight_scale_inv = _create_param_from_subclass_attributes(
            BlockQuantScaleParameter(
                data=weight_scale_inv,
                output_dim=0,
                input_dim=1,
                weight_loader=layer.weight_scale_inv.weight_loader,
            )
        )

    else:
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        # # If using w8a8, torch._scaled_mm needs per tensor, so
        # # requantize the logical shards as a single weight.
        if not self.use_marlin:
            # Dequant -> Quant with max scale so we can run per tensor.

            weight_scale, weight = requantize_with_max_scale(
                weight=weight,
                weight_scale=weight_scale,
                logical_widths=layer.logical_widths,
            )

        weight = self._maybe_pad_weight(weight)
        # Update layer with new values.
        # layer.weight = Parameter(weight.t(), requires_grad=False)
        # layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        layer.weight = _create_param_from_subclass_attributes(
            ModelWeightParameter(
                data=weight,
                output_dim=0,
                input_dim=1,
                weight_loader=layer.weight.weight_loader,
            )
        )
        layer.weight_scale = _create_param_from_subclass_attributes(
            PerTensorScaleParameter(
                data=weight_scale.unsqueeze(0).expand(len(layer.logical_widths)),
                weight_loader=layer.weight_scale.weight_loader,
            )
        )
        # print(f"xueh after {layer.weight.shape=}")
        # print(f"xueh after {layer.weight_scale.shape=}")


def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale

    if self.use_marlin:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias)

    if self.block_quant:
        assert self.quant_config.weight_block_size is not None
        return torch.ops.vllm.apply_w8a8_block_fp8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            use_aiter_and_is_supported=self.use_aiter_and_is_supported,
        )

    weight_scale, weight = requantize_with_max_scale(
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            logical_widths=layer.logical_widths,
    )
    return self.fp8_linear.apply(input=x,
                                    weight=weight.t(),
                                    weight_scale=weight_scale,
                                    out_dtype=self.out_dtype,
                                    input_scale=layer.input_scale,
                                    bias=bias)

def apply_vllm_fp8_patches(block_quant=True):
    print("xueh apply_vllm_fp8_patches")
    if block_quant:
        func_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
        patcher = patch(func_path, process_weights_after_loading)
        patcher.start()
    else:
        func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
        patcher1 = patch(func1_path, process_weights_after_loading)
        patcher1.start()
        func2_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.apply"
        patcher2 = patch(func2_path, apply)
        patcher2.start()


def apply_sglang_fp8_patches(block_quant=True):
    if block_quant:
        print("xueh apply_sglang_fp8_patches")
        func_path = "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
        patcher = patch(func_path, process_weights_after_loading)
        patcher.start()
