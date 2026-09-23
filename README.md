# CRITIC 与熵权法：指标赋权学习示例

本项目提供指标正向化、熵权法、CRITIC 法及综合评分的教学实现，用于学习计算流程与输入边界。代码为独立学习示例，不包含真实挑战杯、其他获奖项目或商业项目的原始源码。

`examples/sample_data.csv` 是社区养老服务评价的纯模拟教学数据，共 6 个虚构对象、6 个指标；不来自真实调研，不包含真实社区或个人信息。示例排名只用于解释计算过程，不能直接作为养老服务质量结论。

## 快速开始

建议 Python 3.10 及以上，在仓库根目录执行；使用独立虚拟环境安装依赖：

```bash
python -m venv .venv
# Windows PowerShell：.\.venv\Scripts\Activate.ps1
# macOS / Linux：source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/demo.py
python -m unittest discover -s tests -v
```

依赖为 NumPy 与 pandas；已验证环境为 Python 3.12、NumPy 2.3、pandas 3.0。

## 计算定义

矩阵的行是评价对象，列是指标。先统一指标方向，再按当前输入样本的最小值、最大值归一化。

- 效益型（越大越好）：`x = (v - min) / (max - min)`。
- 成本型（越小越好）：`x = (max - v) / (max - min)`；支持原始数值为负的指标。
- 常数列：归一化值设为 0，不提供区分信息，两种方法均赋零权重。

**熵权法**：对每个非常数指标计算 `p_ij = x_ij / sum_i(x_ij)`，`e_j = -sum_i(p_ij * ln(p_ij)) / ln(n)`，`w_j = (1 - e_j) / sum_j(1 - e_j)`。约定 `0 * ln(0) = 0`；常数列的熵记为 1。全常数或数值精度下没有可辨别信息时抛出 `ValueError`。

**CRITIC 法**：只对非常数指标计算样本标准差 `s_j`（`ddof=1`）与 Pearson 相关系数 `r_jk`。信息量为 `C_j = s_j * sum_k(1 - r_jk)`，权重为 `C_j / sum_j(C_j)`。保留相关系数的正负号，不使用绝对相关系数。常数列不参与相关计算。少于两个非常数指标，或所有有效指标完全正相关、没有可辨别冲突时，CRITIC 无法按此定义赋权，明确报错；不会静默替换为等权。

**组合权重**：返回两套独立的组合结果。

- 几何平均：`g_j = sqrt(w_entropy_j * w_critic_j)`，再归一化。两种方法无共同正权重指标时，无法计算几何组合，报错。
- 线性组合：`l_j = alpha * w_entropy_j + (1 - alpha) * w_critic_j`，默认 `alpha=0.5`，要求在 `[0, 1]` 内；alpha 仅影响线性组合。

几何平均与“直接相乘后归一化”不同。组合方案是教学选项，不保证比单一方法更合理；两种赋权方法也不会自动确定指标体系是否科学。

**评分**：`score_i = sum_j(x_ij * w_j)`。按指标列名对齐，得分位于 `[0, 1]`；它是相对评价结果，不是概率。改变样本集合会改变归一化范围、权重和排名，跨批次比较须另行固定评价标准。

## 输入约定与边界

- `normalize`、`entropy_weights`、`critic_weights` 至少需要 2 个评价对象；单行样本不足以估计这些客观权重。`score` 可以对已归一化的单行数据评分。
- 只接收有限实数，不自动填补缺失值、丢弃行或转换数值字符串。NaN、无穷、非数值、空矩阵、重复列名均报错。
- `benefit` 与 `cost` 不得重复或重叠，且必须完整覆盖输入的指标列；ID 和名称列须先剔除。
- 直接调用赋权或评分函数时，矩阵须在 `[0, 1]` 范围内。调用方负责完成正确的指标方向转换。
- 外部权重须非负、有限、总和为 1、列名唯一且完整匹配；列顺序可以不同。
- 成本型只表示单调的“越小越好”；区间最优、中间值最优等指标不适用当前正向化函数。

## 示例指标

| 列名 | 教学含义 | 类型 |
|---|---|---|
| bed_density | 每百名老人床位数 | 效益型 |
| staff_ratio | 医护人员配比 | 效益型 |
| coverage | 设施覆盖率 (%) | 效益型 |
| wait_days | 平均等待天数 | 成本型 |
| self_pay | 自费比例 (%) | 成本型 |
| satisfaction | 满意度评分 | 效益型 |

```python
import pandas as pd
from src.weight import normalize, entropy_weights, critic_weights, combined_weights, score

df = pd.read_csv("examples/sample_data.csv").set_index("community")
X = normalize(
    df,
    benefit=["bed_density", "staff_ratio", "coverage", "satisfaction"],
    cost=["wait_days", "self_pay"],
)
w_entropy, _ = entropy_weights(X)
w_critic, _ = critic_weights(X)
w_geom, w_linear = combined_weights(w_entropy, w_critic, alpha=0.5)
print(score(X, w_linear).sort_values(ascending=False))
```

## 目录与许可状态

```text
src/weight.py              核心函数及输入校验
examples/demo.py           可直接运行的教学示例
examples/sample_data.csv   纯模拟数据
tests/test_weight.py       数学结果与异常输入测试
requirements.txt          运行依赖
```

仓库尚未指定许可证。
