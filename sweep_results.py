import importlib
# Force reimport of the module
importlib.reload(importlib.import_module("nanoquant.investigate"))
importlib.reload(importlib.import_module("smoothquant.fake_quant"))

from nanoquant.investigate import sweep, report_sweep, Investigation

repo_dir = "."
save_dir = "."
short_model_name = "opt-125m"

opt_125m_model_path = "/state/partition1/user/zzhang1/cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/"
opt_6_7b_model_path = "/state/partition1/user/zzhang1/cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/"
opt_13b_model_path = "/state/partition1/user/zzhang1/cache/huggingface/hub/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/"

sweep(short_model_name, repo_dir, save_dir, perp=True, local_model_path=opt_6_7b_model_path, local_files_only=True)
