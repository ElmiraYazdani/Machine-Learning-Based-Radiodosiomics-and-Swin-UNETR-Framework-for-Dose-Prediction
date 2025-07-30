import os
import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    Compose,
)
from monai.data import Dataset, DataLoader, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
import argparse
import re


def load_model(checkpoint_path, device, args):
    model = SwinUNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
    ).double()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v.double()
        else:
            new_state_dict[k] = v.double()

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_data(data_dir, json_list, roi_size):
    datalist_json = os.path.join(data_dir, json_list)
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            ToTensord(keys=["image", "label"], dtype=torch.double),
        ]
    )
    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    return val_loader


def save_nifti(data, output_path, affine):
    data = data.astype(np.float64)
    nib_img = nib.Nifti1Image(data, affine)
    nib.save(nib_img, output_path)


def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return match.group(0)
    else:
        return '0'


def main():
    parser = argparse.ArgumentParser(description="Swin UNETR Inference Script")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--json_list", required=True, help="Dataset json file")
    parser.add_argument("--output_dir", required=True, help="Output directory for NIfTI files")
    parser.add_argument("--roi_x", default=96, type=int, help="ROI size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="ROI size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="ROI size in z direction")
    parser.add_argument("--in_channels", default=2, type=int, help="Number of input channels")
    parser.add_argument("--out_channels", default=1, type=int, help="Number of output channels")
    parser.add_argument("--feature_size", default=48, type=int, help="Feature size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="Drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing to save memory")
    parser.add_argument("--sw_batch_size", default=4, type=int, help="Number of sliding window batch size")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="Sliding window inference overlap")

    args = parser.parse_args()

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args)
    val_loader = prepare_data(args.data_dir, args.json_list, (args.roi_x, args.roi_y, args.roi_z))

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device).double()
            affines = batch["image_meta_dict"]["affine"]
            filenames = batch["image_meta_dict"]["filename_or_obj"]
            for idx, image in enumerate(images):
                pred = sliding_window_inference(
                    inputs=image.unsqueeze(0),
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.infer_overlap,
                )
                pred = pred.squeeze().cpu().numpy()
                number = extract_number(os.path.basename(filenames[idx]))
                output_path = os.path.join(args.output_dir, f"DVK{number}.nii.gz")
                save_nifti(pred, output_path, affines[idx])
                print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

