"""社区养老服务能力综合评价示例。

运行（在仓库根目录）：python examples/demo.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.weight import (  # noqa: E402
    combined_weights,
    critic_weights,
    entropy_weights,
    normalize,
    score,
)

BENEFIT = ["bed_density", "staff_ratio", "coverage", "satisfaction"]
COST = ["wait_days", "self_pay"]


def main():
    df = pd.read_csv(os.path.join(ROOT, "examples", "sample_data.csv"))
    names = df["community"]
    data = df.drop(columns="community")

    X = normalize(data, BENEFIT, COST)
    w_ent, entropy = entropy_weights(X)
    w_cri, info = critic_weights(X)
    w_geom, w_lin = combined_weights(w_ent, w_cri)

    weights = pd.DataFrame(
        {
            "entropy": w_ent,
            "critic": w_cri,
            "combined_geom": w_geom,
            "combined_linear": w_lin,
        }
    )
    print("=== 指标权重 ===")
    print(weights.round(4).to_string())

    result = pd.DataFrame(
        {
            "community": names,
            "score_geom": score(X, w_geom),
            "score_linear": score(X, w_lin),
        }
    ).sort_values("score_linear", ascending=False)
    print("\n=== 综合得分（降序）===")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
