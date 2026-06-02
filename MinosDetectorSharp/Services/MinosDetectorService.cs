using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.DotNet;
using MinosDetectorSharp.Models;

namespace MinosDetectorSharp.Services;

public class MinosDetectorService : IDisposable
{
    private const int MaxTokens = 8192;
    private static readonly string[] Labels = ["Non-refusal", "Refusal"];

    private readonly InferenceSession _session;
    private readonly Tokenizer _tokenizer;

    public MinosDetectorService(string modelDir, int? intraOpNumThreads = null, int? interOpNumThreads = null)
    {
        var onnxFile = Path.Combine(modelDir, "model.onnx");
        if (!File.Exists(onnxFile))
            throw new FileNotFoundException($"ONNX model not found: {onnxFile}");

        var tokenizerFile = Path.Combine(modelDir, "tokenizer.json");
        if (!File.Exists(tokenizerFile))
            throw new FileNotFoundException($"Tokenizer not found: {tokenizerFile}");

        var sessionOptions = new SessionOptions
        {
            IntraOpNumThreads = intraOpNumThreads ?? Math.Max(1, Environment.ProcessorCount - 1),
            InterOpNumThreads = interOpNumThreads ?? 1
        };

        _session = new InferenceSession(onnxFile, sessionOptions);
        _tokenizer = new Tokenizer(vocabPath: tokenizerFile);
    }

    public string FormatInput(string userMessage, string assistantResponse)
    {
        return $"<|user|>\n{userMessage}\n<|assistant|>\n{assistantResponse}";
    }

    public string FormatMultiTurnInput((string User, string Assistant)[] turns)
    {
        var parts = new string[turns.Length];
        for (int i = 0; i < turns.Length; i++)
        {
            parts[i] = $"<|user|>\n{turns[i].User}\n<|assistant|>\n{turns[i].Assistant}";
        }
        return string.Join("\n", parts);
    }

    public DetectionResult Detect(string text)
    {
        var allIds = _tokenizer.Encode(text).Select(id => (long)id).ToArray();
        var inputIds = allIds.Length <= MaxTokens
            ? allIds
            : allIds.Take(MaxTokens).ToArray();

        return PredictRaw(inputIds);
    }

    public DetectionResult Detect(string userMessage, string assistantResponse)
    {
        return Detect(FormatInput(userMessage, assistantResponse));
    }

    public DetectionResult DetectMultiTurn(params (string User, string Assistant)[] turns)
    {
        return Detect(FormatMultiTurnInput(turns));
    }

    private DetectionResult PredictRaw(long[] inputIds)
    {
        var seqLen = inputIds.Length;
        var attentionMask = Enumerable.Repeat(1L, seqLen).ToArray();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, [1, seqLen])),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(attentionMask, [1, seqLen]))
        };

        using var results = _session.Run(inputs);
        var logits = results.First().AsEnumerable<float>().ToArray();

        // Softmax
        var maxLogit = logits.Max();
        var exps = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
        var sumExps = exps.Sum();
        var probs = exps.Select(x => (float)(x / sumExps)).ToArray();

        var labelId = Array.IndexOf(probs, probs.Max());

        return new DetectionResult(
            Label: Labels[labelId],
            Confidence: probs[labelId],
            Logits: logits,
            Probabilities: probs
        );
    }

    public void Dispose()
    {
        _session?.Dispose();
        _tokenizer?.Dispose();
    }
}
