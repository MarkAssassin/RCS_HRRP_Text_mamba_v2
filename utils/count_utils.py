import copy

from thop import profile
from thop import clever_format


def compute_flops_params(model, input_list):
    """
    计算模型的FLOPs和参数量。

    对 model 做 deepcopy 后再交给 thop，避免 profile 在原模型上
    残留 total_ops/total_params 与 forward hook，导致后续训练报错
    （如 'ReLU' object has no attribute 'total_ops'）。

    参数:
        model: 需要计算的PyTorch模型
        input_list: 输入模型的张量列表，形状应与模型输入匹配

    返回:
        flops: 格式化后的FLOPs字符串
        params: 格式化后的参数量字符串
    """
    model_for_count = copy.deepcopy(model)
    model_for_count.eval()
    flops, params = profile(model_for_count, inputs=tuple(input_list), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"计算量FLOPS为:{flops}, 参数量Params为:{params}")
    del model_for_count
    return flops, params
