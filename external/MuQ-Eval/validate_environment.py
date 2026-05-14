"""
Pre-execution environment validation for MuQ-MOS experiments.

Checks all prerequisites before launching the ablation chain:
  - Python version, CUDA, GPU VRAM
  - Required packages (correct versions)
  - Model weights cached on HuggingFace Hub
  - Datasets accessible
  - Disk space
  - Config file integrity
  - Smoke test (single forward pass)

Usage:
    python validate_environment.py [--skip-smoke-test] [--skip-download-check]

Exit code 0 = all checks pass, 1 = blocking failure detected.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "torch": "2.1.0",
    "transformers": "4.38.0",
    "accelerate": "0.25.0",
    "peft": "0.7.0",
    "torchaudio": "2.1.0",
    "librosa": "0.10.0",
    "soundfile": "0.12.0",
    "datasets": "2.16.0",
    "scipy": "1.11.0",
    "sklearn": "1.3.0",
    "numpy": "1.24.0",
    "pandas": "2.0.0",
    "omegaconf": "2.3.0",
    "yaml": "6.0",
    "tqdm": "4.60.0",
}

HF_MODELS = [
    "OpenMuQ/MuQ-large-msd-iter",
    "m-a-p/MERT-v1-95M",
]

HF_DATASETS = [
    "BAAI/MusicEval",
    "ASLP-lab/SongEval",
]

CONFIG_FILES = [
    "configs/base.yaml",
    "configs/A1_frozen_mlp.yaml",
    "configs/A2_frozen_ordinal.yaml",
    "configs/A3a_lora.yaml",
    "configs/A3b_contrastive.yaml",
    "configs/A3_full.yaml",
    "configs/A6_mert.yaml",
]

MIN_DISK_GB = 100
MIN_VRAM_GB = 14


class ValidationResult:
    def __init__(self):
        self.checks = []
        self.blocking_failures = 0
        self.warnings = 0

    def ok(self, name, detail=""):
        self.checks.append(("PASS", name, detail))
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))

    def fail(self, name, detail=""):
        self.blocking_failures += 1
        self.checks.append(("FAIL", name, detail))
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))

    def warn(self, name, detail=""):
        self.warnings += 1
        self.checks.append(("WARN", name, detail))
        print(f"  [WARN] {name}" + (f" -- {detail}" if detail else ""))

    def summary(self):
        total = len(self.checks)
        passed = sum(1 for s, _, _ in self.checks if s == "PASS")
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY: {passed}/{total} passed, "
              f"{self.blocking_failures} failures, {self.warnings} warnings")
        if self.blocking_failures == 0:
            print("STATUS: READY FOR EXECUTION")
        else:
            print("STATUS: BLOCKED -- fix failures before running experiments")
        print(f"{'='*60}")
        return self.blocking_failures == 0


def check_python(v: ValidationResult):
    """Check Python version >= 3.10."""
    print("\n--- Python Environment ---")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        v.ok(f"Python {major}.{minor}")
    else:
        v.fail(f"Python {major}.{minor}", "Requires >= 3.10")


def check_cuda(v: ValidationResult):
    """Check CUDA availability and GPU VRAM."""
    print("\n--- GPU / CUDA ---")
    try:
        import torch
        if torch.cuda.is_available():
            v.ok("CUDA available")
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            v.ok(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
            if vram_gb >= MIN_VRAM_GB:
                v.ok(f"VRAM >= {MIN_VRAM_GB} GB")
            else:
                v.warn(f"VRAM {vram_gb:.1f} GB < {MIN_VRAM_GB} GB",
                       "A3 full and A6 may need reduced batch size")

            # Check bf16 support
            if torch.cuda.get_device_capability(0) >= (8, 0):
                v.ok("bf16 supported (compute capability >= 8.0)")
            else:
                v.warn("bf16 not supported", "Will fall back to fp16")
        else:
            v.fail("CUDA not available", "GPU required for training")
    except ImportError:
        v.fail("PyTorch not installed")


def check_packages(v: ValidationResult):
    """Check required packages are installed with minimum versions."""
    print("\n--- Required Packages ---")
    from packaging import version as pkg_version

    for pkg_name, min_ver in REQUIRED_PACKAGES.items():
        import_name = pkg_name
        if pkg_name == "sklearn":
            import_name = "sklearn"
        elif pkg_name == "yaml":
            import_name = "yaml"

        try:
            mod = importlib.import_module(import_name)
            installed_ver = getattr(mod, "__version__", "unknown")
            if installed_ver != "unknown":
                try:
                    if pkg_version.parse(installed_ver) >= pkg_version.parse(min_ver):
                        v.ok(f"{pkg_name} {installed_ver}")
                    else:
                        v.fail(f"{pkg_name} {installed_ver}", f"Requires >= {min_ver}")
                except Exception:
                    v.ok(f"{pkg_name} {installed_ver} (version parse skipped)")
            else:
                v.ok(f"{pkg_name} (installed, version unknown)")
        except ImportError:
            v.fail(f"{pkg_name} NOT INSTALLED", f"pip install {pkg_name}>={min_ver}")


def check_hf_models(v: ValidationResult, skip_download: bool):
    """Check HuggingFace model weights are cached."""
    print("\n--- HuggingFace Model Weights ---")
    if skip_download:
        v.warn("Download check skipped (--skip-download-check)")
        return

    try:
        from huggingface_hub import scan_cache_dir, HfApi
        api = HfApi()
        for model_id in HF_MODELS:
            try:
                info = api.model_info(model_id)
                v.ok(f"{model_id} exists on Hub ({info.modelId})")
            except Exception as e:
                v.warn(f"{model_id} not accessible", str(e))
    except ImportError:
        v.warn("huggingface_hub not installed", "Cannot verify model cache")


def check_hf_datasets(v: ValidationResult, skip_download: bool):
    """Check HuggingFace datasets are accessible."""
    print("\n--- HuggingFace Datasets ---")
    if skip_download:
        v.warn("Download check skipped (--skip-download-check)")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for ds_id in HF_DATASETS:
            try:
                info = api.dataset_info(ds_id)
                v.ok(f"{ds_id} accessible ({info.id})")
            except Exception as e:
                v.warn(f"{ds_id} not accessible", str(e))
    except ImportError:
        v.warn("huggingface_hub not installed", "Cannot verify dataset access")


def check_disk_space(v: ValidationResult):
    """Check available disk space."""
    print("\n--- Disk Space ---")
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / 1e9
    if free_gb >= MIN_DISK_GB:
        v.ok(f"Disk: {free_gb:.1f} GB free (>= {MIN_DISK_GB} GB)")
    else:
        v.warn(f"Disk: {free_gb:.1f} GB free (< {MIN_DISK_GB} GB)",
               "May run out during training/checkpoints")


def check_configs(v: ValidationResult):
    """Validate all experiment config files exist and parse correctly."""
    print("\n--- Experiment Configs ---")
    try:
        from omegaconf import OmegaConf
        for cfg_path in CONFIG_FILES:
            p = Path(cfg_path)
            if p.exists():
                try:
                    OmegaConf.load(str(p))
                    v.ok(f"{cfg_path}")
                except Exception as e:
                    v.fail(f"{cfg_path} parse error", str(e))
            else:
                v.fail(f"{cfg_path} NOT FOUND")
    except ImportError:
        v.fail("omegaconf not installed")


def check_source_modules(v: ValidationResult):
    """Verify all source modules can be imported."""
    print("\n--- Source Modules ---")
    modules = [
        "src.data",
        "src.encoders",
        "src.model",
        "src.losses",
        "src.trainer",
        "src.baselines",
        "src.evaluation",
    ]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            v.ok(f"{mod_name}")
        except Exception as e:
            v.fail(f"{mod_name} import error", str(e)[:100])


def smoke_test(v: ValidationResult, skip: bool):
    """Run a minimal forward pass to verify model + data pipeline."""
    print("\n--- Smoke Test (single forward pass) ---")
    if skip:
        v.warn("Smoke test skipped (--skip-smoke-test)")
        return

    try:
        import torch
        from omegaconf import OmegaConf, ListConfig

        cfg = OmegaConf.load("configs/A1_frozen_mlp.yaml")
        if "defaults" in cfg:
            base_name = cfg.defaults[0] if isinstance(cfg.defaults, (list, ListConfig)) else cfg.defaults
            base_path = Path("configs") / f"{base_name}.yaml"
            if base_path.exists():
                base_cfg = OmegaConf.load(str(base_path))
                cfg = OmegaConf.merge(base_cfg, cfg)

        # Create dummy input
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy_audio = torch.randn(2, 240000).to(device)  # 10s at 24kHz

        from src.model import MusicQualityModel
        model = MusicQualityModel(cfg).to(device)
        model.eval()

        with torch.no_grad():
            output = model(dummy_audio)

        if isinstance(output, dict) and "MI" in output:
            v.ok(f"Forward pass OK, output keys: {list(output.keys())}, "
                 f"MI shape: {output['MI'].shape}")
        else:
            v.warn("Forward pass returned unexpected format", str(type(output)))

        # Memory check
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
            v.ok(f"Peak GPU memory (batch=2): {mem_mb:.0f} MB")

        del model, dummy_audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        v.fail("Smoke test failed", str(e)[:200])


def main():
    parser = argparse.ArgumentParser(description="Validate experiment environment")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--skip-download-check", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("MuQ-MOS Experiment Environment Validation")
    print("=" * 60)

    v = ValidationResult()

    check_python(v)
    check_cuda(v)
    check_packages(v)
    check_hf_models(v, args.skip_download_check)
    check_hf_datasets(v, args.skip_download_check)
    check_disk_space(v)
    check_configs(v)
    check_source_modules(v)
    smoke_test(v, args.skip_smoke_test)

    ready = v.summary()

    # Write validation report
    report = {
        "ready": ready,
        "checks": [{"status": s, "name": n, "detail": d} for s, n, d in v.checks],
        "blocking_failures": v.blocking_failures,
        "warnings": v.warnings,
    }
    report_path = Path("outputs") / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved to: {report_path}")

    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
