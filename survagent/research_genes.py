"""Assay statistics from Gen_Agent/Gene_describe.py."""

def calculate_Genetic_Mutation_situation(Gen_cnv: dict) -> str:
    import numpy as np
    from scipy.stats import entropy

    value_list = list(Gen_cnv.values())

    # 计算均值
    mean = sum(value_list) / len(value_list)
    # 计算中位数
    middle = np.percentile(value_list, 50)
    # 计算突变占比情况
    increase_num = 0
    reduce_num = 0

    for single_value in value_list:
        if single_value > 0:
            increase_num += 1
        elif single_value < 0:
            reduce_num += 1

    increase_rate = increase_num / len(value_list)
    reduce_rate = reduce_num / len(value_list)

    # 计算样本标准差
    sample_std = np.std(value_list, ddof=1)

    # 计算熵
    unique, counts = np.unique(value_list, return_counts=True)
    probabilities = counts / counts.sum()
    entropy_value = entropy(probabilities, base=2)

    # 描述
    describe = f"""Gene Copy Number Variation (CNV) Statistics:\nMean:{mean:.4f}\nMedian:{middle:.4f}\nVariation Proportion:\n    - Proportion of genes with amplification:{increase_rate:.4f}\n    - Proportion of genes with deletion:{reduce_rate:.4f}"""

    return describe


def calculate_Expression_level_situation(Gen_rnaseq: dict) -> str:
    import numpy as np

    value_list = list(Gen_rnaseq.values())

    # 计算均值
    mean = sum(value_list) / len(value_list)
    # 计算中位数
    middle = np.percentile(value_list, 50)
    # 计算突变占比情况
    increase_num = 0
    reduce_num = 0

    for single_value in value_list:
        if single_value > 0:
            increase_num += 1
        elif single_value < 0:
            reduce_num += 1

    increase_rate = increase_num / len(value_list)
    reduce_rate = reduce_num / len(value_list)

    # 计算样本标准差
    sample_std = np.std(value_list, ddof=1)

    # 描述
    describe = f"""RNA-seq Characterization Statistics:\nMean:{mean:.4f}\nMedian:{middle:.4f}\nDysregulation Proportion:\n    - Proportion of upregulated genes:{increase_rate:.4f}\n    - Proportion of downregulated genes:{reduce_rate:.4f}"""

    return describe
