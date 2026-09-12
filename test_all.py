from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ============================================================
# 路径设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "comparison.yaml"
DATA_ROOT = PROJECT_ROOT / "DHG2016"

# 你截图中的模型目录
MODEL_ROOT = PROJECT_ROOT / "_smoke_compare_runs"

EVALUATE_SCRIPT = PROJECT_ROOT / "evaluate_compare.py"
PLOT_SCRIPT = PROJECT_ROOT / "plot_comparison.py"

# 需要测试的7个修补模型
MODEL_NAMES = [
    "anatomy_mae",
    "gru_ae",
    "motion_mae",
    "repairformer",
    "stgcn_ae",
    "tcn_ae",
    "transformer_mae",
]

# 支持识别的检查点后缀
CHECKPOINT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
}


def find_checkpoints(model_dir: Path) -> list[Path]:
    """递归查找模型目录中的检查点文件。"""
    checkpoints = []

    if not model_dir.exists():
        return checkpoints

    for path in model_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES:
            checkpoints.append(path)

    return sorted(checkpoints)


def check_required_paths() -> None:
    """检查评估所需的文件和目录。"""
    required_paths = {
        "配置文件": CONFIG_PATH,
        "数据集目录": DATA_ROOT,
        "模型根目录": MODEL_ROOT,
        "评估脚本": EVALUATE_SCRIPT,
    }

    errors = []

    for description, path in required_paths.items():
        if not path.exists():
            errors.append(
                f"找不到{description}：{path}"
            )

    if errors:
        print("\n路径检查失败：")

        for error in errors:
            print("  -", error)

        raise FileNotFoundError(
            "请先修改脚本顶部的路径设置。"
        )


def inspect_models() -> dict[str, list[Path]]:
    """检查每个模型目录中是否存在模型检查点。"""
    print("\n" + "=" * 72)
    print("检查已经训练的模型")
    print("=" * 72)

    available_models: dict[str, list[Path]] = {}
    missing_models: list[str] = []

    for model_name in MODEL_NAMES:
        model_dir = MODEL_ROOT / model_name
        checkpoints = find_checkpoints(model_dir)

        print(f"\n模型：{model_name}")
        print(f"目录：{model_dir}")

        if not model_dir.exists():
            print("状态：模型目录不存在")
            missing_models.append(model_name)
            continue

        if not checkpoints:
            print("状态：目录存在，但没有找到 .pt/.pth/.ckpt 文件")
            missing_models.append(model_name)
            continue

        print(f"状态：找到 {len(checkpoints)} 个检查点")

        for checkpoint in checkpoints:
            relative_path = checkpoint.relative_to(PROJECT_ROOT)
            size_mb = checkpoint.stat().st_size / 1024 / 1024

            print(
                f"  - {relative_path} "
                f"({size_mb:.2f} MB)"
            )

        available_models[model_name] = checkpoints

    print("\n" + "-" * 72)
    print(f"可测试模型数量：{len(available_models)}")
    print(f"缺少检查点数量：{len(missing_models)}")

    if missing_models:
        print(
            "缺少检查点的模型：",
            ", ".join(missing_models),
        )

    if not available_models:
        raise RuntimeError(
            "\n没有找到任何模型检查点。\n"
            "请展开模型目录，确认其中存在 .pt、.pth 或 .ckpt 文件。"
        )

    return available_models


def inspect_shared_classifier() -> None:
    """检查共享分类器是否存在。"""
    classifier_dir = MODEL_ROOT / "shared_classifier"
    checkpoints = find_checkpoints(classifier_dir)

    print("\n" + "=" * 72)
    print("检查共享分类器")
    print("=" * 72)

    if not classifier_dir.exists():
        print(f"共享分类器目录不存在：{classifier_dir}")
        print("分类准确率指标可能无法计算。")
        return

    if not checkpoints:
        print("共享分类器目录存在，但没有找到检查点。")
        print("Gesture/Finger/Fine Accuracy 可能无法计算。")
        return

    print(f"找到 {len(checkpoints)} 个共享分类器检查点：")

    for checkpoint in checkpoints:
        relative_path = checkpoint.relative_to(PROJECT_ROOT)
        print("  -", relative_path)


def run_command(command: list[str]) -> None:
    """运行命令并实时显示终端输出。"""
    print("\n" + "=" * 72)
    print("开始执行命令")
    print("=" * 72)
    print(" ".join(command))
    print()

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def evaluate_models() -> None:
    """调用项目原有评估程序测试全部模型。"""
    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--config",
        str(CONFIG_PATH),
        "--data-root",
        str(DATA_ROOT),
        "--methods",
        "all",
    ]

    run_command(command)


def generate_plots() -> None:
    """生成模型对比图。"""
    if not PLOT_SCRIPT.exists():
        print(
            f"\n没有找到绘图脚本：{PLOT_SCRIPT}"
        )
        print("模型评估已经完成，但不会生成对比图。")
        return

    command = [
        sys.executable,
        str(PLOT_SCRIPT),
    ]

    run_command(command)


def main() -> None:
    print("=" * 72)
    print("7个骨架修补模型常规测试")
    print("=" * 72)

    print(f"项目目录：{PROJECT_ROOT}")
    print(f"Python解释器：{sys.executable}")
    print(f"配置文件：{CONFIG_PATH}")
    print(f"数据集目录：{DATA_ROOT}")
    print(f"模型目录：{MODEL_ROOT}")

    # 1. 检查路径
    check_required_paths()

    # 2. 检查7个模型检查点
    available_models = inspect_models()

    # 3. 检查共享分类器
    inspect_shared_classifier()

    print("\n准备测试以下模型：")

    for index, model_name in enumerate(
        available_models,
        start=1,
    ):
        print(f"{index}. {model_name}")

    # 4. 执行正式评估
    evaluate_models()

    # 5. 生成结果图
    generate_plots()

    print("\n" + "=" * 72)
    print("模型测试完成")
    print("=" * 72)

    print(
        "\n请查看以下目录：\n"
        f"{MODEL_ROOT / 'results'}"
    )


if __name__ == "__main__":
    try:
        main()

    except subprocess.CalledProcessError as error:
        print("\n" + "=" * 72)
        print("评估程序运行失败")
        print("=" * 72)

        print(f"返回代码：{error.returncode}")
        print(f"失败命令：{error.cmd}")

        print(
            "\n可能的原因：\n"
            "1. comparison.yaml 中的模型根目录不是 "
            "_smoke_compare_runs；\n"
            "2. 检查点文件名与 evaluate_compare.py "
            "要求的不一致；\n"
            "3. 模型参数与检查点参数不一致；\n"
            "4. shared_classifier 检查点不存在；\n"
            "5. 数据集路径或测试集划分不正确。"
        )

        input("\n按 Enter 键退出……")
        raise

    except Exception as error:
        print("\n" + "=" * 72)
        print("测试失败")
        print("=" * 72)
        print(error)

        input("\n按 Enter 键退出……")
        raise