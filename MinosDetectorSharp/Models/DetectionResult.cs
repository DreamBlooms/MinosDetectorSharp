namespace MinosDetectorSharp.Models;

public record DetectionResult(
    string Label,
    float Confidence,
    float[] Logits,
    float[] Probabilities
);
