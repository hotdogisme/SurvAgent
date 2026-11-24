"""Algorithms retained from get_wsi_feature_path_run.py and split_tile.py."""

def deduplicate_by_cosine(feature_dict, threshold=0.9):
    """
    基于余弦相似度去重。
    
    参数:
        feature_dict (dict): key为样本名，value为2048维向量 (np.ndarray)
        threshold (float): 相似度阈值，大于该值的两个样本认为重复。
    
    返回:
        retained_keys (list): 去重后保留下来的key
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Step 1: 将 dict 转成矩阵形式
    keys = list(feature_dict.keys())
    features = np.stack([feature_dict[k] for k in keys], axis=0)  # shape: (N, 2048)
    
    # Step 2: 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(features)
    
    # Step 3: 去重逻辑
    n = len(keys)
    retained = np.ones(n, dtype=bool)  # True表示保留
    
    for i in range(n):
        if not retained[i]:
            continue
        # 对于后续的样本，如果与当前样本相似度过高，则去除
        for j in range(i + 1, n):
            if retained[j] and sim_matrix[i, j] > threshold:
                retained[j] = False
    
    # Step 4: 返回保留的key
    retained_keys = [keys[i] for i in range(n) if retained[i]]
    return retained_keys


def merge_wsi_descriptions(all_img_description, backend, extract_prompt, combine_prompt):
    """Apply the archived four-report extraction and recursive merge."""
    chunk_size = 4
    split_4_group = [
        all_img_description[i : i + chunk_size]
        for i in range(0, len(all_img_description), chunk_size)
    ]
    json_report = []
    for description_group in split_4_group:
        description_all = ""
        for single_description_id, single_description in enumerate(description_group):
            description_all += f"Slide {single_description_id+1}: {single_description}\n"
        extract_feature_factors_prompt = extract_prompt.replace("{current_description}", description_all)
        json_report.append(backend.chat(extract_feature_factors_prompt))

    final_report = json_report
    while len(final_report) > 1:
        split_4_group = [final_report[i : i + chunk_size] for i in range(0, len(final_report), chunk_size)]
        final_report = []
        for description_group in split_4_group:
            description_all = ""
            for single_description_id, single_description in enumerate(description_group):
                description_all += f"Slide {single_description_id+1}: {single_description}\n\n"
            combine_report_prompt = combine_prompt.replace("{comprehensive_description}", description_all)
            final_report.append(backend.chat(combine_report_prompt))
    return final_report[0], json_report
