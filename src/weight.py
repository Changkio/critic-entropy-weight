"""指标赋权教学实现：熵权法、CRITIC 法及组合权重。

行表示评价对象，列表示指标。赋权前须正向化并归一化至 [0, 1]。
无信息或 CRITIC 无冲突时明确报错，不静默生成等权结果。
"""
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


def _matrix(df, *, min_rows=2, normalized=True):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("指标矩阵必须是 pandas DataFrame。")
    if len(df) < min_rows or df.shape[1] == 0:
        raise ValueError(f"至少需要 {min_rows} 个评价对象和 1 个指标。")
    if not df.columns.is_unique:
        raise ValueError("指标列名不能重复。")
    if any(not is_numeric_dtype(t) or is_bool_dtype(t) or is_complex_dtype(t)
           for t in df.dtypes):
        raise ValueError("指标必须为实数数值，不能包含字符串、布尔值或复数。")
    if df.isna().any().any():
        raise ValueError("指标包含缺失值；请先明确缺失值处理方案。")
    result = df.astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("指标不能包含无穷值。")
    if normalized and ((result < 0).any().any() or (result > 1).any().any()):
        raise ValueError("请先使用 normalize 将指标归一化至 [0, 1]。")
    return result


def _weights(w):
    if not isinstance(w, pd.Series) or w.empty or not w.index.is_unique:
        raise ValueError("权重必须是非空、指标名唯一的 pandas Series。")
    if not is_numeric_dtype(w.dtype) or is_bool_dtype(w.dtype) or is_complex_dtype(w.dtype):
        raise ValueError("权重必须为实数数值。")
    if w.isna().any():
        raise ValueError("权重不能包含缺失值。")
    result = w.astype(float)
    if not np.isfinite(result.to_numpy()).all() or (result < 0).any():
        raise ValueError("权重必须为有限非负值。")
    if not np.isclose(result.sum(), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("权重之和必须为 1。")
    return result


def normalize(df: pd.DataFrame, benefit: list, cost: list) -> pd.DataFrame:
    """min-max 正向化；所有指标须且只能指定一种方向，常数列置 0。"""
    data = _matrix(df, normalized=False)
    benefit, cost = list(benefit), list(cost)
    names = benefit + cost
    if len(names) != len(set(names)):
        raise ValueError("效益型与成本型指标不能重复或重叠。")
    if set(names) != set(data.columns):
        raise ValueError("benefit 与 cost 必须恰好覆盖所有指标列。")
    result = pd.DataFrame(index=data.index)
    for name in names:
        lo, hi = data[name].min(), data[name].max()
        span = hi - lo
        if not np.isfinite(span):
            raise ValueError("指标范围发生数值溢出，请先缩放数据。")
        if span == 0:
            result[name] = 0.0
        elif name in benefit:
            result[name] = (data[name] - lo) / span
        else:
            result[name] = (hi - data[name]) / span
    return result[names]


def entropy_weights(X: pd.DataFrame):
    """返回 (熵权, 熵值)；常数列的熵约定为 1，权重为 0。"""
    X = _matrix(X)
    varying = X.max() > X.min()
    if not varying.any():
        raise ValueError("所有指标均为常数，无法计算有区分度的熵权。")
    active = X.loc[:, varying]
    P = active.div(active.sum(axis=0), axis=1)
    # 按极限定义 0 * log(0) = 0，避免用缺失值自动跳过异常数据。
    logs = np.log(P.where(P > 0, 1.0))
    E = pd.Series(1.0, index=X.columns)
    E.loc[varying] = (-(P * logs).sum(axis=0) / np.log(len(X))).clip(0.0, 1.0)
    information = 1.0 - E
    if information.sum() <= np.finfo(float).eps:
        raise ValueError("熵信息差异不足，无法稳定赋权。")
    return information / information.sum(), E


def critic_weights(X: pd.DataFrame):
    """返回 (CRITIC 权重, 信息量)，使用样本标准差和 Pearson 相关。"""
    X = _matrix(X)
    varying = X.max() > X.min()
    active = X.loc[:, varying]
    if active.shape[1] < 2:
        raise ValueError("CRITIC 至少需要 2 个非常数指标才能计算指标间冲突。")
    # 常数列不参与相关矩阵，避免未定义的相关系数被当成零相关。
    corr = active.corr().clip(-1.0, 1.0)
    if not np.isfinite(corr.to_numpy()).all():
        raise ValueError("相关系数计算不稳定，请检查指标数值范围。")
    corr_array = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_array, 1.0)
    conflict = pd.Series((1.0 - corr_array).sum(axis=0), index=active.columns)
    C = pd.Series(0.0, index=X.columns)
    C.loc[varying] = active.std(axis=0, ddof=1) * conflict
    # 浮点计算可能把完全正相关算成 1 - 1e-16；不能放大该舍入误差。
    if conflict.max() <= 1e-12 or C.sum() <= 0:
        raise ValueError("指标间没有可辨别的冲突，CRITIC 信息量为 0，无法赋权。")
    return C / C.sum(), C


def combined_weights(w_entropy: pd.Series, w_critic: pd.Series, alpha: float = 0.5):
    """返回 (归一化几何平均, 线性组合)；alpha 为线性组合中的熵权占比。"""
    w_entropy, w_critic = _weights(w_entropy), _weights(w_critic)
    if set(w_entropy.index) != set(w_critic.index):
        raise ValueError("两组权重必须覆盖相同的指标。")
    w_critic = w_critic.reindex(w_entropy.index)
    if not isinstance(alpha, Real) or isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha 必须是 [0, 1] 内的实数。")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha 必须是 [0, 1] 内的实数。")
    geom = np.sqrt(w_entropy) * np.sqrt(w_critic)
    if geom.sum() == 0:
        raise ValueError("两组权重没有共同的正权重指标，几何组合无法归一化。")
    geom = geom / geom.sum()
    linear = alpha * w_entropy + (1.0 - alpha) * w_critic
    return geom, linear / linear.sum()


def score(X: pd.DataFrame, w: pd.Series) -> pd.Series:
    """按列名对齐计算得分；权重必须完整覆盖全部指标。"""
    X, w = _matrix(X, min_rows=1), _weights(w)
    if set(X.columns) != set(w.index):
        raise ValueError("权重指标必须与矩阵的全部指标一致。")
    return X.mul(w.reindex(X.columns), axis=1).sum(axis=1)
