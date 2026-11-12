using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace ArenaMind.Play
{
    /// <summary>
    /// The trained network, run by Unity's Sentis (package com.unity.ai.inference)
    /// from Resources/policy.onnx: observation and sliders in, 18 scores out.
    /// The network is tiny (about 25 000 weights), so the CPU runs it in well
    /// under a millisecond and there is nothing to gain from the GPU.
    /// </summary>
    public sealed class ArenaPolicy : IDisposable
    {
        private readonly Worker worker;
        private readonly TensorShape shape;

        public ArenaPolicy(int inputs)
        {
            var asset = Resources.Load<ModelAsset>("policy");
            if (asset == null)
            {
                throw new InvalidOperationException(
                    "Resources/policy.onnx is missing: run  python -m tools.export_unity <model.pt>  first");
            }
            Model model = ModelLoader.Load(asset);
            worker = new Worker(model, BackendType.CPU);
            shape = new TensorShape(1, inputs);
        }

        /// <summary>The 18 scores for one input (move 9, turn 5, shoot 2, dash 2).</summary>
        public float[] Scores(float[] input)
        {
            using (var tensor = new Tensor<float>(shape, input))
            {
                worker.Schedule(tensor);
                var output = worker.PeekOutput() as Tensor<float>;
                return output.DownloadToArray();
            }
        }

        /// <summary>
        /// Runs the check cases exported with the network and returns the
        /// largest difference from the scores PyTorch gave for them. Anything
        /// above about 1e-4 means the export or the inputs are wrong.
        /// </summary>
        public float Check(ArenaData data)
        {
            float worst = 0f;
            foreach (ArenaCheck check in data.checks)
            {
                float[] scores = Scores(check.input);
                if (scores.Length != check.logits.Length)
                {
                    return float.PositiveInfinity;
                }
                for (int i = 0; i < scores.Length; i++)
                {
                    worst = Mathf.Max(worst, Mathf.Abs(scores[i] - check.logits[i]));
                }
            }
            return worst;
        }

        public void Dispose()
        {
            worker.Dispose();
        }
    }
}
