using System;
using UnityEngine;

namespace ArenaMind.Play
{
    // What tools/export_unity.py writes next to the network, in Resources/arenamind.json.
    // Field names are those of the JSON file.

    [Serializable]
    public class ArenaLevel
    {
        public string name;  // L1 (easiest) to L5 (the policy at full strength)
        public double every;  // the policy decides every this many steps
        public float[] elo;  // its rating with each personality, 0 if not rated
    }

    [Serializable]
    public class ArenaPersonality
    {
        public string name;
        public string settings;  // as on the Python command line, e.g. aggression=-1
        public float[] sliders;  // extra inputs of the network
    }

    [Serializable]
    public class ArenaCheck
    {
        public float[] input;
        public float[] logits;  // what PyTorch gives for this input
    }

    [Serializable]
    public class ArenaPair
    {
        public string stronger, weaker;  // two neighbouring levels
        public double actual;  // the stronger one's share of the points against the weaker in Python
        public double noise;  // its standard error
    }

    [Serializable]
    public class ArenaData
    {
        public string checkpoint;
        public int inputs;  // observation and sliders
        public int observation;
        public int[] actionSizes;
        public string[] sliders;
        public ArenaPersonality[] personalities;
        public ArenaLevel[] levels;
        public ArenaCheck[] checks;
        public ArenaPair[] pairs;  // how the levels did against each other in Python

        public static ArenaData Load()
        {
            var text = Resources.Load<TextAsset>("arenamind");
            if (text == null)
            {
                throw new InvalidOperationException(
                    "Resources/arenamind.json is missing: run  python -m tools.export_unity <model.pt>  first");
            }
            return JsonUtility.FromJson<ArenaData>(text.text);
        }
    }
}
