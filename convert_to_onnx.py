#!/usr/bin/env python3
"""
Minos-v1 (ModernBERT) ONNX Conversion Script
Uses HuggingFace Optimum to export ModernBERT to ONNX format.
"""

import argparse
import os
import sys
import shutil
import json
import numpy as np


def convert_to_onnx(model_dir: str, output_dir: str):
    print("=" * 60)
    print("Minos-v1 (ModernBERT) ONNX Conversion Tool")
    print("=" * 60)

    # Load config to verify model type
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    model_type = config.get("model_type", "unknown")
    print(f"\n[1/5] Model type: {model_type}")
    assert model_type == "modernbert", f"Expected modernbert, got {model_type}"

    # Export using Optimum
    print(f"\n[2/5] Loading model and exporting to ONNX...")
    from optimum.onnxruntime import ORTModelForSequenceClassification

    model = ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        export=True,
    )
    print(f"       ONNX export completed.")

    # Save the exported model
    print(f"\n[3/5] Saving ONNX model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # Also save tokenizer files to output dir
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"       Tokenizer saved to: {output_dir}")

    # List output files
    print(f"\n[4/5] Output files:")
    for f_name in sorted(os.listdir(output_dir)):
        f_path = os.path.join(output_dir, f_name)
        if os.path.isfile(f_path):
            size_mb = os.path.getsize(f_path) / (1024 * 1024)
            print(f"       {f_name} ({size_mb:.2f} MB)")

    # Verify the ONNX model
    print(f"\n[5/5] Verifying ONNX model...")
    try:
        import onnx
        import onnxruntime as ort
        from transformers import AutoTokenizer as Tok

        onnx_files = [f for f in os.listdir(output_dir) if f.endswith(".onnx")]
        onnx_path = os.path.join(output_dir, onnx_files[0])

        onnx.checker.check_model(onnx.load(onnx_path))
        print("       ONNX model structure check passed!")

        # Runtime verification
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        tok = Tok.from_pretrained(output_dir)

        test_text = "<|user|>\nWhat is the capital of France?\n<|assistant|>\nThe capital of France is Paris."
        inputs = tok(test_text, return_tensors="np")

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        outputs = session.run(None, onnx_inputs)
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        label_id = int(np.argmax(probs))
        labels = {0: "Non-refusal", 1: "Refusal"}
        print(f"       Test input: {test_text[:60]}...")
        print(f"       Prediction: {labels[label_id]} (confidence: {probs[label_id]:.4f})")
        print(f"       Logits: {logits.tolist()}")
        print(f"       Probabilities: {probs.tolist()}")
        print("       ONNX runtime verification passed!")

    except Exception as e:
        print(f"       Verification error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert Minos-v1 ModernBERT to ONNX")
    parser.add_argument(
        "-m", "--model",
        default=os.path.join(os.path.dirname(__file__), "Minos-v1"),
        help="Model directory (default: Minos-v1 in script directory)"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(__file__), "Minos-v1-onnx"),
        help="Output ONNX directory (default: Minos-v1-onnx in script directory)"
    )
    args = parser.parse_args()

    convert_to_onnx(args.model, args.output)


if __name__ == "__main__":
    main()
