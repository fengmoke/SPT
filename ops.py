from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
from utils import *
from main import get_argparser


def count_parameters(model, trainable_only=False):
    """
    统计模型参数量（单位：百万）
    :param trainable_only: 是否只统计可训练参数
    :return: 参数字典，包含各子模块参数量统计
    """
    stats = {}
    total = 0
    trainable = 0
    
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters() if not trainable_only or p.requires_grad)
        if params > 0:  # 只记录有参数的模块
            stats[name] = {
                "parameters(M)": params / 1e6,
                "trainable": any(p.requires_grad for p in module.parameters())
            }
            total += params
            if any(p.requires_grad for p in module.parameters()):
                trainable += params

    return {
        "total(M)": total / 1e6,
        "trainable(M)": trainable / 1e6,
        "modules": stats
    }

args = args = get_argparser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = built_model(args)
print(model)
model.to(device)

# 新增参数量统计
param_stats = count_parameters(model, False)
print(f"Total Parameters: {param_stats['total(M)']:.2f}M")
print(f"Trainable Parameters: {param_stats['trainable(M)']:.2f}M")

# 详细模块统计（可选）
for name, stats in param_stats['modules'].items():
    print(f"{name}: {stats['parameters(M)']:.2f}M ({'trainable' if stats['trainable'] else 'fixed'})")