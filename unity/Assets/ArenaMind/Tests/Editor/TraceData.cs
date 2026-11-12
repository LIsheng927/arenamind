using System;
using System.IO;
using UnityEngine;

namespace ArenaMind.Tests
{
    // The games and checks recorded by tools/export_traces.py (python -m tools.export_traces).
    // Field names are those of the JSON file. Arrays are flat: one step after another.

    [Serializable]
    public class TraceConfig
    {
        public double dt, maxTime, size, spawnJitter, radius, maxHp, speed;
        public double bulletSpeed, bulletRadius, bulletLifetime, damage, cooldown;
        public double dashMultiplier, dashDuration, dashCooldown, rayRange;
        public int maxBullets, rays, trackedBullets, obsDim, maxSteps;
        public double[] obstacles, spawns, turnDegrees;
        public int[] actionSizes;
    }

    [Serializable]
    public class TraceGame
    {
        public string name;
        public int steps;
        public double[] startPos, startAngle;  // x0, y0, x1, y1 and the two facings
        public float[] startObs;  // both players' observations before the first step
        // After each step:
        public int[] actions;  // 8 per step: player 0's move, turn, shoot, dash, then player 1's
        public double[] pos, angle, vel, hp;  // 4, 2, 4, 2 per step
        public int[] timers;  // 6 per step: shoot cooldowns, dash cooldowns, dash steps left
        public int[] bullets;  // 6 per step: alive (1) or not, owner * 3 + slot
        public double[] bulletPos;  // 12 per step: x, y of each bullet (0 if not alive)
        public int[] visible, unseen;  // 1 and 2 per step
        public double[] lastSeen;  // 4 per step
        public int[] done, outcome;  // 1 each per step; outcome for player 0
        public float[] obs;  // 98 per step: both players' observations
    }

    [Serializable]
    public class TraceGate
    {
        public double every;
        public int newGameAt;
        public int[] decide;  // 1 on the steps the policy decides
    }

    [Serializable]
    public class TraceControls
    {
        public double[] wrapIn, wrapOut;
        public int[] turnDegrees, turnsForDegrees;
        public double[] turnDiffs;
        public int[] turns;
        public int[] moveEighths, movesForEighths;
        public double[] moveAngles;
        public int[] moves;
    }

    [Serializable]
    public class TraceFile
    {
        public TraceConfig config;
        public TraceGame[] games;
        public TraceGate[] gates;
        public TraceControls controls;

        private static TraceFile loaded;

        public static string FilePath
        {
            get { return Path.Combine(Application.dataPath, "ArenaMind", "Tests", "Editor", "traces.json"); }
        }

        public static TraceFile Load()
        {
            if (loaded == null)
            {
                if (!File.Exists(FilePath))
                {
                    throw new FileNotFoundException(
                        "no recorded games; run  python -m tools.export_traces  in the Python project", FilePath);
                }
                loaded = JsonUtility.FromJson<TraceFile>(File.ReadAllText(FilePath));
            }
            return loaded;
        }
    }
}
