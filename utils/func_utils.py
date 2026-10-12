# 计算网络模型的计算量和参数量
from thop import profile
from thop import clever_format # 这里不使用thop库，因为去获取整个model 的mac，
                                 # 所以会导致total_params以及total_ops被注册到网络中，
                                 # 又因为该pt文件中根本没有这个key，所以才会导致miss错误

def compute_flops_params(model, input_list):
    """
    计算模型的FLOPs和参数量

    参数:
    model: 需要计算的PyTorch模型
    input_list: 输入模型的张量列表，形状应与模型的输入匹配

    返回:
    flops: 模型的FLOPs
    params: 模型的参数量
    """
    flops, params = profile(model, inputs=input_list) # 由于每个 MAC 包含一个乘法和一个加法，所以 FLOPs 将是 MAC 计算量的两倍。
    flops, params = clever_format([flops, params], "%.3f")
    print(f"计算量FLOPS为:{flops}, 参数量Params为:{params}")
    return flops, params