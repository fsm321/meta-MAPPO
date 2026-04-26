# evaluate_parallel.py
# 并行训练调用的评估接口，直接复用 evaluate.py，避免两套评估逻辑不一致

from evaluate import (
    normalize_obs,
    evaluate_policy,
    evaluate_combat_metrics,
    evaluate_robustness,
    evaluate_failure_recovery,
)


if __name__ == "__main__":
    import runpy
    runpy.run_module("evaluate", run_name="__main__")