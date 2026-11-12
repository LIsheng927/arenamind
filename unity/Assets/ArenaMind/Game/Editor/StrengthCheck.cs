using System;
using System.Globalization;
using System.IO;
using System.Text;
using ArenaMind.Sim;
using UnityEditor;
using UnityEngine;

namespace ArenaMind.Play
{
    /// <summary>
    /// Menu ArenaMind > Play the levels against each other.
    ///
    /// The tests check the pieces one by one: the game rules and observations
    /// against recorded Python games, the network against PyTorch's scores.
    /// This checks the whole NPC at once: each difficulty level plays the next
    /// weaker one in Unity (C# game, Sentis network, C# sampling and decision
    /// gaps), as many games as tools/difficulty_ladder.py played in Python, and
    /// the stronger level's share of the points is compared with Python's.
    ///
    /// Fixed before running: a pair agrees if the two shares are within three
    /// standard errors (z between -3 and 3). The games are random, so the
    /// shares never match exactly; a mistake in the port, such as a wrong
    /// observation number, would make a level clearly weaker than in Python.
    ///
    /// Writes runs/unity_check/levels.csv in the Python project and prints a
    /// table to the Console.
    /// </summary>
    public static class StrengthCheck
    {
        private const int GamesPerSide = 200;  // as --check-games of tools/difficulty_ladder.py
        private const double Agreement = 3.0;  // largest |z| for a pair to agree

        [MenuItem("ArenaMind/Play the levels against each other")]
        public static void Run()
        {
            ArenaData data = ArenaData.Load();
            var cfg = new GameConfig();
            float[] neutral = new float[data.inputs - cfg.ObsDim];  // the policy's own play, as in the ladder
            var csv = new StringBuilder("stronger,weaker,games,unity,unity_noise,python,python_noise,z,agrees\n");
            var report = new StringBuilder();
            int pairs = 0, agreed = 0;
            long decisions = 0;
            var clock = System.Diagnostics.Stopwatch.StartNew();
            using (var policy = new ArenaPolicy(data.inputs))
            {
                try
                {
                    for (int k = data.levels.Length - 1; k > 0; k--)
                    {
                        ArenaLevel strong = data.levels[k], weak = data.levels[k - 1];
                        double points = 0.0;
                        int games = 0;
                        for (int side = 0; side < 2; side++)  // the stronger level plays each side half the games
                        {
                            for (int g = 0; g < GamesPerSide; g++)
                            {
                                float done = ((data.levels.Length - 1 - k) * 2 * GamesPerSide + side * GamesPerSide + g)
                                             / (float)((data.levels.Length - 1) * 2 * GamesPerSide);
                                if (EditorUtility.DisplayCancelableProgressBar(
                                        "ArenaMind", strong.name + " against " + weak.name + ", game " + (games + 1), done))
                                {
                                    throw new OperationCanceledException();
                                }
                                var random = new System.Random(100000 * k + 1000 * side + g);
                                var game = new Game(cfg);
                                game.Reset(random);
                                var a = new NpcBrain(policy, cfg, side, data.inputs, strong.every, neutral, random);
                                var b = new NpcBrain(policy, cfg, 1 - side, data.inputs, weak.every, neutral, random);
                                while (!game.Done)
                                {
                                    PlayerAction x = a.Act(game), y = b.Act(game);
                                    if (side == 0) game.Step(x, y);
                                    else game.Step(y, x);
                                }
                                decisions += a.Decisions + b.Decisions;
                                int outcome = side == 0 ? game.Outcome : -game.Outcome;  // for the stronger level
                                points += outcome > 0 ? 1.0 : (outcome == 0 ? 0.5 : 0.0);
                                games++;
                            }
                        }

                        double unity = points / games;
                        double unityNoise = Math.Sqrt(unity * (1 - unity) / games);
                        ArenaPair python = Find(data, strong.name, weak.name);
                        string line = strong.name + " vs " + weak.name + ": Unity " + Percent(unity) + " +- "
                                      + Percent(unityNoise);
                        string row = strong.name + "," + weak.name + "," + games + "," + Number(unity) + ","
                                     + Number(unityNoise);
                        if (python != null)
                        {
                            double spread = Math.Sqrt(unityNoise * unityNoise + python.noise * python.noise);
                            double z = spread > 0 ? (unity - python.actual) / spread : (unity == python.actual ? 0.0 : 99.0);
                            bool agrees = Math.Abs(z) <= Agreement;
                            pairs++;
                            agreed += agrees ? 1 : 0;
                            line += ", Python " + Percent(python.actual) + " +- " + Percent(python.noise) + ", z = "
                                    + z.ToString("0.00", CultureInfo.InvariantCulture) + (agrees ? "  agrees" : "  DIFFERS");
                            row += "," + Number(python.actual) + "," + Number(python.noise) + ","
                                   + Number(z) + "," + (agrees ? "yes" : "no");
                        }
                        else
                        {
                            line += ", no Python result to compare with";
                            row += ",,,,";
                        }
                        report.AppendLine(line);
                        csv.AppendLine(row);
                    }
                }
                catch (OperationCanceledException)
                {
                    Debug.Log("ArenaMind: strength check cancelled");
                    return;
                }
                finally
                {
                    EditorUtility.ClearProgressBar();
                }
            }

            double seconds = clock.Elapsed.TotalSeconds;
            string path = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "runs", "unity_check", "levels.csv"));
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, csv.ToString());
            Debug.Log("ArenaMind: the levels against each other, " + 2 * GamesPerSide + " games per pair\n" + report
                      + agreed + " of " + pairs + " pairs agree with Python (|z| <= " + Agreement + "). "
                      + decisions + " decisions in " + seconds.ToString("0.0") + " s ("
                      + (1000.0 * seconds / Math.Max(decisions, 1)).ToString("0.000") + " ms per decision, game included).\n"
                      + "Wrote " + path);
        }

        private static ArenaPair Find(ArenaData data, string stronger, string weaker)
        {
            if (data.pairs == null)
            {
                return null;
            }
            foreach (ArenaPair pair in data.pairs)
            {
                if (pair.stronger == stronger && pair.weaker == weaker)
                {
                    return pair;
                }
            }
            return null;
        }

        private static string Percent(double share)
        {
            return (100 * share).ToString("0.0", CultureInfo.InvariantCulture) + "%";
        }

        private static string Number(double value)
        {
            return value.ToString("0.0000", CultureInfo.InvariantCulture);
        }
    }
}
