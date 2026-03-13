%%writefile infer_colab.py
import os
import sys
import argparse
import torch
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import time
import hashlib
import shutil
from tqdm import tqdm

sys.path.append('/content/Histoformer')
# ================= ARCHITECTURE IMPORT =================
try:
    from basicsr.models.archs.histoformer_arch import Histoformer
except ImportError:
    print("❌ ERROR: Could not import Histoformer. Make sure the basicsr folder is in your Colab working directory.")
    sys.exit(1)
# =======================================================

WINDOW_SIZE = 192

def tta8_forward(model, x):
    preds = []
    for k in range(8):
        x_aug = x.clone()
        if k >= 4: x_aug = x_aug.transpose(2, 3)
        if k % 4 >= 2: x_aug = torch.flip(x_aug, [2])
        if k % 2 == 1: x_aug = torch.flip(x_aug, [3])
        out_aug = model(x_aug)
        out_aug = torch.clamp(out_aug, 0.0, 1.0)
        if k % 2 == 1: out_aug = torch.flip(out_aug, [3])
        if k % 4 >= 2: out_aug = torch.flip(out_aug, [2])
        if k >= 4: out_aug = out_aug.transpose(2, 3)
        preds.append(out_aug)
    return torch.median(torch.stack(preds), dim=0)[0]

def tta4_forward(model, x):
    preds = []
    preds.append(torch.clamp(model(x), 0.0, 1.0))

    out1 = model(torch.flip(x, [3]))
    preds.append(torch.flip(torch.clamp(out1, 0.0, 1.0), [3]))

    out2 = model(torch.flip(x, [2]))
    preds.append(torch.flip(torch.clamp(out2, 0.0, 1.0), [2]))

    out3 = model(torch.flip(x, [2, 3]))
    preds.append(torch.flip(torch.clamp(out3, 0.0, 1.0), [2, 3]))
    return torch.median(torch.stack(preds), dim=0)[0]

def tta2_forward(model, x):
    out_orig = model(x)
    x_hflip = torch.flip(x, [3])
    out_hflip = model(x_hflip)
    out_hflip_reversed = torch.flip(out_hflip, [3])
    return (out_orig + out_hflip_reversed) / 2.0

def get_powered_hann_mask(window_size, power, device='cpu'):
    w = torch.hann_window(window_size, periodic=False, dtype=torch.float32, device=device)
    weight_2d = (w.unsqueeze(0) * w.unsqueeze(1)) ** power
    return weight_2d.unsqueeze(0).unsqueeze(0) + 1e-6

def sliding_window_inference(model, img_tensor, window_size, overlap_h, overlap_w, use_jitter, weight_mask, tta_mode, offset_override=None):
    b, c, h, w = img_tensor.shape
    stride_h = window_size - overlap_h
    stride_w = window_size - overlap_w

    if offset_override is not None:
        offset_y, offset_x = offset_override
    else:
        offset_y = np.random.randint(0, stride_h) if use_jitter else 0
        offset_x = np.random.randint(0, stride_w) if use_jitter else 0

    pad_h_bottom = (stride_h - ((h + offset_y - window_size) % stride_h)) % stride_h
    pad_w_right  = (stride_w - ((w + offset_x - window_size) % stride_w)) % stride_w
    padded_img = torch.nn.functional.pad(
        img_tensor, (offset_x, pad_w_right, offset_y, pad_h_bottom), mode='reflect'
    )
    _, _, ph, pw = padded_img.shape
    output_accumulator = torch.zeros_like(padded_img)
    weight_accumulator = torch.zeros_like(padded_img)

    for y in range(0, ph - window_size + 1, stride_h):
        for x in range(0, pw - window_size + 1, stride_w):
            patch = padded_img[:, :, y:y+window_size, x:x+window_size]
            with torch.no_grad():
                if tta_mode == 8: pred_patch = tta8_forward(model, patch)
                elif tta_mode == 4: pred_patch = tta4_forward(model, patch)
                else: pred_patch = tta2_forward(model, patch)
            output_accumulator[:, :, y:y+window_size, x:x+window_size] += pred_patch * weight_mask
            weight_accumulator[:, :, y:y+window_size, x:x+window_size] += weight_mask

    final_padded_output = output_accumulator / weight_accumulator
    final_output = final_padded_output[:, :, offset_y : offset_y + h, offset_x : offset_x + w]
    return torch.clamp(final_output, 0.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description="Histoformer Grid Search Script")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment (e.g., Exp_A_JitterOff)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input validation images")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to 105k checkpoint")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the final zip (e.g., /content/drive/MyDrive/NTIRE_Results)")

    # Tunable Knobs
    parser.add_argument("--tta", type=int, choices=[2, 4, 8], default=2, help="TTA Mode")
    parser.add_argument("--no_jitter", action='store_true', help="Pass this flag to turn JITTER OFF")
    parser.add_argument("--hann", type=float, default=1.6, help="Hann window power")
    parser.add_argument("--overlap_h", type=int, default=64, help="Vertical overlap")
    parser.add_argument("--overlap_w", type=int, default=32, help="Horizontal overlap")
    parser.add_argument("--unsharp_w", type=float, default=0.15, help="Unsharp Mask Weight")
    parser.add_argument("--unsharp_s", type=float, default=1.0, help="Unsharp Mask Sigma")
    parser.add_argument("--dual_offset", action='store_true', help="Use deterministic dual offset averaging")

    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_JITTER = not args.no_jitter

    # Generate the output folder inside the specified output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    OUT_DIR = os.path.join(args.output_dir, f"submission_{args.exp_name}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"🚀 Loading Model: {args.ckpt_path}")
    model = Histoformer(
        inp_channels=3, out_channels=3, dim=36, num_blocks=[4,4,6,8],
        num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.667,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
    ).to(DEVICE)

    checkpoint = torch.load(args.ckpt_path, map_location=DEVICE)
    if "ema_model" in checkpoint: model.load_state_dict(checkpoint["ema_model"])
    elif "params_ema" in checkpoint: model.load_state_dict(checkpoint["params_ema"])
    else: model.load_state_dict(checkpoint.get("model", checkpoint.get("params", checkpoint)))
    model.eval()

    weight_mask = get_powered_hann_mask(WINDOW_SIZE, power=args.hann, device=DEVICE)
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_exts)])

    print(f"📦 Executing Exp: {args.exp_name}")
    print(f"Knobs -> TTA:{args.tta} | Jitter:{USE_JITTER} | Hann:{args.hann} | OH:{args.overlap_h} OW:{args.overlap_w}")
    print(f"Knobs -> Unsharp W:{args.unsharp_w} S:{args.unsharp_s} | DualOffset:{args.dual_offset}")

    total_time = 0.0
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Inference"):
            img_seed = int(hashlib.sha256(filename.encode('utf-8')).hexdigest(), 16) % (2**32)
            np.random.seed(img_seed)

            full_path = os.path.join(args.input_dir, filename)
            img_bgr = cv2.imread(full_path)
            if img_bgr is None: continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            start_t = time.time()

            if args.dual_offset:
                stride_h, stride_w = WINDOW_SIZE - args.overlap_h, WINDOW_SIZE - args.overlap_w
                out1 = sliding_window_inference(model, input_tensor, WINDOW_SIZE, args.overlap_h, args.overlap_w, False, weight_mask, args.tta, offset_override=(0,0))
                out2 = sliding_window_inference(model, input_tensor, WINDOW_SIZE, args.overlap_h, args.overlap_w, False, weight_mask, args.tta, offset_override=(stride_h//2, stride_w//2))
                output = (out1 + out2) / 2.0
            else:
                output = sliding_window_inference(model, input_tensor, WINDOW_SIZE, args.overlap_h, args.overlap_w, USE_JITTER, weight_mask, args.tta)

            # --- MILD UNSHARP MASK ---
            if args.unsharp_w > 0:
                blur = TF.gaussian_blur(output, kernel_size=[5, 5], sigma=[args.unsharp_s, args.unsharp_s])
                output = torch.clamp(output + args.unsharp_w * (output - blur), 0.0, 1.0)

            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            total_time += (time.time() - start_t)

            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(OUT_DIR, filename), output_bgr)

    if len(image_files) > 0:
        avg_time = total_time / len(image_files)
        print(f"\n⏱️ Average Runtime: {avg_time:.4f} seconds/image")
        readme_content = f"runtime per image [s] : {avg_time:.4f}\nCPU[1] / GPU[0] : 0\nExtra Data [1] / No Extra Data [0] : 0\nOther description: {args.exp_name} | TTAx{args.tta}, Hann({args.hann}), OH:{args.overlap_h}, OW:{args.overlap_w}, Jitter:{USE_JITTER}, Unsharp({args.unsharp_w}, sigma {args.unsharp_s}), DualOffset:{args.dual_offset}"
        with open(os.path.join(OUT_DIR, "readme.txt"), "w") as f:
            f.write(readme_content)

    # Save the zip directly to the requested output directory
    zip_base_name = os.path.join(args.output_dir, args.exp_name)
    print(f"\n🗜️ Zipping '{OUT_DIR}' into '{zip_base_name}.zip'...")
    shutil.make_archive(zip_base_name, 'zip', OUT_DIR)

    # Optional cleanup to save Colab disk space:
    # shutil.rmtree(OUT_DIR)

    print(f"✅ Success! File saved as {zip_base_name}.zip")

if __name__ == "__main__":
    main()
