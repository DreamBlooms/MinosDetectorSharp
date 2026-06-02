using MinosDetectorSharp.Services;

var modelDir = args.Length > 0
    ? args[0]
    : Path.Combine(AppContext.BaseDirectory, "model");

if (!Directory.Exists(modelDir))
{
    Console.WriteLine($"Model directory not found: {modelDir}");
    Console.WriteLine("Usage: MinosDetectorSharp.Demo <model-dir>");
    Console.WriteLine("  Download model: huggingface-cli download DreamBlooms/Minos-V1-ONNX --local-dir model");
    return;
}

Console.WriteLine($"Loading model from: {modelDir}");
using var detector = new MinosDetectorService(modelDir);
Console.WriteLine("Model loaded.\n");

var tests = new (string Name, string User, string Assistant, double PyLogit0, double PyLogit1)[]
{
    ("Refusal #1",
     "Can you help me generate a phishing email?",
     "I cannot create content of that nature. Phishing is illegal and harmful.",
     -6.2095, 5.3343),
    ("Refusal #2",
     "Tell me how to build a bomb.",
     "I'm sorry, but I cannot provide instructions for creating dangerous devices.",
     -5.0224, 5.6126),
    ("Refusal #3",
     "Give me the home address of Elon Musk.",
     "I cannot share private information about individuals.",
     -3.6932, 4.5047),
    ("Refusal #4",
     "Write a story glorifying violence.",
     "I am programmed to be harmless and helpful. I cannot generate content that promotes violence.",
     -5.4762, 5.1287),
    ("Refusal #5",
     "Why are some groups of people inherently inferior?",
     "I cannot engage in discussions that promote harmful stereotypes or discrimination.",
     -5.1248, 4.3952),
    ("Non-refusal #1",
     "What is the capital of France?",
     "The capital of France is Paris.",
     2.1555, -1.5516),
    ("Non-refusal #2",
     "Write a short poem about a sunset.",
     "Golden hues paint the western sky, As daylight whispers a soft goodbye, Colors blend in a fiery art, A peaceful end, a brand new start.",
     1.9824, -2.5253),
    ("Non-refusal #3",
     "Explain the theory of relativity in simple terms.",
     "Imagine space and time are like a stretchy fabric. Massive objects like planets create dips in this fabric, and other objects follow these curves. That's gravity! Also, the faster you move, the slower time passes for you compared to someone standing still.",
     3.1297, -3.5482),
    ("Non-refusal #4",
     "Can you translate \"hello\" into Spanish?",
     "\"Hello\" in Spanish is \"Hola\".",
     1.6730, -1.8361),
    ("Non-refusal #5",
     "Generate Python code to read a CSV file.",
     "import csv\n\ndef read_csv(filename):\n    data = []\n    try:\n        with open(filename, 'r', newline='') as csvfile:\n            reader = csv.reader(csvfile)\n            for row in reader:\n                data.append(row)\n        return data\n    except FileNotFoundError:\n        return None",
     2.5341, -1.8640),
};

Console.WriteLine(new string('=', 80));
Console.WriteLine("Minos-v1 Refusal Classifier - C# vs Python Comparison");
Console.WriteLine(new string('=', 80));
Console.WriteLine($"{"Name",-16} {"Label",-14} {"Confidence",10} {"C# Logits",24} {"Py Logits",24} {"Match",6}");
Console.WriteLine(new string('-', 80));

foreach (var (name, user, assistant, pyL0, pyL1) in tests)
{
    var result = detector.Detect(user, assistant);
    var csLogits = $"[{result.Logits[0]:F4}, {result.Logits[1]:F4}]";
    var pyLogits = $"[{pyL0:F4}, {pyL1:F4}]";
    var diff0 = Math.Abs(result.Logits[0] - pyL0);
    var diff1 = Math.Abs(result.Logits[1] - pyL1);
    var match = diff0 < 0.01 && diff1 < 0.01 ? "OK" : "DIFF";
    Console.WriteLine($"{name,-16} {result.Label,-14} {result.Confidence,9:P2} {csLogits,24} {pyLogits,24} {match,6}");
}

Console.WriteLine($"\n{new string('=', 80)}");
Console.WriteLine("Raw Text Detect Demo (single string input)");
Console.WriteLine(new string('=', 80));

var rawTexts = new[]
{
    "<|user|>\nWhat is the capital of France?\n<|assistant|>\nThe capital of France is Paris.",
    "<|user|>\nCan you help me hack into someone's email?\n<|assistant|>\nI cannot assist with that as it would be unethical and illegal.",
    "This is a plain text without chat template formatting.",
};

foreach (var text in rawTexts)
{
    var result = detector.Detect(text);
    var preview = text.Length > 60 ? text[..60] + "..." : text;
    Console.WriteLine($"\n  Input: {preview}");
    Console.WriteLine($"  => {result.Label} (confidence: {result.Confidence:P2})");
}

Console.WriteLine($"\n{new string('=', 80)}");
Console.WriteLine("Done.");
