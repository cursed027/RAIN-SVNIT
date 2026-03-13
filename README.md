```markdown
# 🌧️ NTIRE 2026: Day & Night Raindrop Removal (Dual-Focused Images)
**Team:** RAIN-SVNIT
**Official Submission for the NTIRE 2026 Challenge at CVPR**

This repository contains the inference code and evaluation setup for our submission to the NTIRE 2026 Day and Night Raindrop Removal Challenge. Our approach utilizes a heavily analyzed **Histoformer** architecture, leveraging a strict 105k-iteration early-stopping plateau, Test-Time Augmentation (TTA), and high-frequency unsharp masking to effectively remove raindrops while strictly preserving the natural depth-of-field defocus blur inherent to dual-focused images.

---

## 🛠️ 1. Environment Setup

The code is built on PyTorch and the BasicSR framework. To replicate our environment, install the required dependencies:

```bash
# Core dependencies
pip install torch torchvision

# Framework & Processing dependencies
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips

```

**Crucial BasicSR Path Fix:**
Ensure the `basicsr` module is properly recognized by Python before running inference:

```bash
touch basicsr/__init__.py
export PYTHONPATH=$(pwd):$PYTHONPATH

```

---

## 📦 2. Data & Weights Preparation

### Download the Champion Model Weight

Our optimal checkpoint (`105,000` iterations) avoids the late-stage "blur valley" common in dual-degradation tasks. Download it directly to your working directory:

```bash
gdown --id "1XpzXHyZVVdeqE4_QdzN5Wgm5ZnObLy_5" -O net_g_105000.pth

```

### Setup the Dataset

Download and extract the competition testing or validation datasets:

```bash
# Example: Download Test Data
gdown --id "1yM2FwraaLx3Ql4CpvYqLYwM-DoONmC5e" -O RainDrop.zip
unzip RainDrop.zip -d ./test_data

```

---

## 🚀 3. One-Line Evaluation (`eval.py`)

We have condensed our inference pipeline, including the sliding-window attention mechanism, TTA (Flips/Rotations), and Unsharp Masking, into a single highly configurable script.

Run the following command to execute inference on the dataset. This reproduces our exact final Codalab submission:

```bash
python eval.py \
  --exp_name Final_Submission_105k \
  --input_dir ./test_data/RainDrop/ \
  --ckpt_path ./net_g_105000.pth \
  --output_dir ./NTIRE_Results \
  --tta 2 \
  --hann 1.4 \
  --overlap_h 64 \
  --overlap_w 32 \
  --unsharp_w 0.15 \
  --unsharp_s 1.0 

```

### ⚙️ Inference Flags Explained

* `--tta 2`: Enables Test-Time Augmentation (horizontal flip + reverse) to stabilize predictions.
* `--hann 1.4`: Controls the power of the Hann window mask for seamless patch blending.
* `--overlap_h 64` / `--overlap_w 32`: Defines the overlapping pixel density for the sliding window to prevent edge artifacts.
* `--unsharp_w 0.15`: Applies a mild unsharp mask weight to recover high-frequency structural edges.
* `--no_jitter`: (Optional) Disables random grid shifting during sliding window inference.

---

## 📂 Output

The script will automatically process all images, apply the restoration pipeline, calculate the average runtime per image, and generate a final `.zip` file in the `--output_dir` (e.g., `Final_Submission_105k.zip`). This zip file includes the required `readme.txt` with runtime metrics and is ready for immediate Codalab submission.

## 📄 Citation

If you utilize this pipeline or our dataset methodology, please acknowledge the original Histoformer architecture:

```bibtex
@InProceedings{sun2024restoring,
    author="Sun, Shangquan and Ren, Wenqi and Gao, Xinwei and Wang, Rui and Cao, Xiaochun",
    title="Restoring Images in Adverse Weather Conditions via Histogram Transformer",
    booktitle="Computer Vision -- ECCV 2024",
    year="2025",
    pages="111--129"
}

```

```

Once you push this to GitHub, your code is perfectly reproducible. Do you want to start structuring the abstract and introduction for the CVPR workshop paper now?

```
