"""
Training Script — Fine-tune YOLOv8 on infrastructure defect dataset.

Usage:
    python train.py --data data/dataset.yaml --epochs 100 --img 640
    python train.py --mode val --weights models/best.pt
"""

import argparse
import os
from ultralytics import YOLO


def train(args):
    """Train YOLOv8 on the defect detection dataset."""
    print("=" * 60)
    print("  Infrastructure Defect Detection — Training")
    print("=" * 60)

    # Load base model
    model = YOLO(args.model)
    print(f"Base model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.img}")
    print(f"Batch size: {args.batch}")
    print("=" * 60)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        name="defect_detection",
        project="runs/train",
        patience=20,
        save=True,
        save_period=10,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        augment=True,
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # Copy best weights
    best_weights = "runs/train/defect_detection/weights/best.pt"
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        import shutil
        shutil.copy(best_weights, "models/best.pt")
        print(f"\n✅ Best weights saved to models/best.pt")

    print("\n📊 Training Results:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    return results


def validate(args):
    """Validate model on the dataset."""
    print("Validating model...")

    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        imgsz=args.img,
        batch=args.batch,
        device=args.device,
    )

    print(f"\n📊 Validation Results:")
    print(f"   mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"   Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"   Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A')}")

    return results


def export_model(args):
    """Export model to ONNX for deployment."""
    print("Exporting model...")

    model = YOLO(args.weights)
    model.export(format="onnx", imgsz=args.img, simplify=True)
    print("✅ Model exported to ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Defect Detector")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "export"])
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Weights for val/export")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="", help="cuda device or cpu")
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "val":
        validate(args)
    elif args.mode == "export":
        export_model(args)
