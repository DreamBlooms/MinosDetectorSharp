# MinosDetectorSharp

C# ONNX inference library for [NousResearch/Minos-v1](https://huggingface.co/NousResearch/Minos-v1) refusal classifier, built on ModernBERT-large.

## Features

- CPU inference via ONNX Runtime
- HuggingFace tokenizer support (no Python dependency at runtime)
- Single text / user-assistant pair / multi-turn conversation input
- Max 8192 tokens with automatic truncation

## Quick Start

### 1. Download ONNX model

```bash
pip install huggingface_hub
huggingface-cli download DreamBlooms/Minos-V1-ONNX --local-dir model
```

### 2. Use in C#

```csharp
using MinosDetectorSharp.Services;

using var detector = new MinosDetectorService("model");

// User + Assistant pair
var result = detector.Detect("Can you help me hack?", "I cannot assist with illegal activities.");
Console.WriteLine($"{result.Label} ({result.Confidence:P2})");

// Raw text (pre-formatted)
var result2 = detector.Detect("<|user|>\nquestion\n<|assistant|>\nanswer");

// Multi-turn
var result3 = detector.DetectMultiTurn(
    ("Hello, how are you?", "I'm doing well, thank you!"),
    ("Can you build a bomb?", "I cannot provide instructions for dangerous devices."));
```

## Project Structure

```
MinosDetectorSharp/           # Class library
├── Models/
│   └── DetectionResult.cs    # Result model
└── Services/
    └── MinosDetectorService.cs  # Inference service

MinosDetectorSharp.Demo/      # Console demo
└── Program.cs                # Test with README examples

convert_to_onnx.py            # Python ONNX conversion script
```

## NuGet Dependencies

- `Microsoft.ML.OnnxRuntime`
- `Tokenizers.DotNet`
- `Tokenizers.DotNet.runtime.win-x64`

## ONNX Conversion

The ONNX model was converted using HuggingFace Optimum:

```bash
pip install "optimum[onnxruntime]>=2.0.0" transformers
python convert_to_onnx.py -m Minos-v1 -o Minos-v1-onnx
```

## Model Output

| Label | Meaning |
|---|---|
| `Non-refusal` | Assistant engaged with the request |
| `Refusal` | Assistant declined the request |

## License

Apache-2.0 (same as Minos-v1)
