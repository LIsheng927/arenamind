using System;
using ArenaMind.Sim;
using NUnit.Framework;

namespace ArenaMind.Tests
{
    /// <summary>
    /// The C# game against the Python original: every recorded game is
    /// replayed from the same start with the same actions, and the state and
    /// both observations are compared after every step.
    /// </summary>
    public class SimTests
    {
        // The two versions do the same arithmetic in the same order, so they
        // agree to the last few bits; sin and cos of different maths libraries
        // may differ in the last bit, which is what these allow for.
        private const double Tolerance = 1e-9;
        private const double VelocityTolerance = 1e-7;  // a position difference divided by 0.1 s
        private const double ObservationTolerance = 1e-5;  // float32 numbers of size ~1

        private static readonly string[] SelfNames =
            { "hp", "shoot cooldown", "dash cooldown", "velocity forward", "velocity right",
              "x", "y", "cos facing", "sin facing", "time left" };
        private static readonly string[] OpponentNames =
            { "opponent visible", "opponent forward", "opponent right", "opponent distance", "opponent hp",
              "opponent velocity forward", "opponent velocity right", "time since seen" };
        private static readonly string[] BulletNames =
            { "present", "forward", "right", "velocity forward", "velocity right" };

        private static string NameOf(int k, GameConfig cfg)
        {
            if (k < 10) return SelfNames[k];
            if (k < 18) return OpponentNames[k - 10];
            if (k < 18 + cfg.Rays) return "ray " + (k - 18);
            int b = k - 18 - cfg.Rays;
            return "bullet " + (b / 5) + " " + BulletNames[b % 5];
        }

        private static void Near(double actual, double expected, double tolerance, string at, string what)
        {
            if (!(Math.Abs(actual - expected) <= tolerance))
            {
                Assert.Fail(at + ": " + what + " is " + actual.ToString("R") + ", Python had " + expected.ToString("R"));
            }
        }

        private static void Same(int actual, int expected, string at, string what)
        {
            if (actual != expected)
            {
                Assert.Fail(at + ": " + what + " is " + actual + ", Python had " + expected);
            }
        }

        private static void CheckObservations(Game game, float[] expected, int start, float[] buffer, string at)
        {
            int n = game.Config.ObsDim;
            Observation.Write(game, 0, buffer, 0);
            Observation.Write(game, 1, buffer, n);
            for (int k = 0; k < 2 * n; k++)
            {
                Near(buffer[k], expected[start + k], ObservationTolerance, at,
                     "observation of player " + (k / n) + ", number " + (k % n) + " (" + NameOf(k % n, game.Config) + ")");
            }
        }

        [Test]
        public void ConfigMatchesPython()
        {
            TraceConfig c = TraceFile.Load().config;
            var cfg = new GameConfig();
            double[] python = { c.dt, c.maxTime, c.size, c.spawnJitter, c.radius, c.maxHp, c.speed, c.bulletSpeed,
                                c.bulletRadius, c.bulletLifetime, c.damage, c.cooldown, c.dashMultiplier,
                                c.dashDuration, c.dashCooldown, c.rayRange };
            double[] ours = { cfg.Dt, cfg.MaxTime, cfg.Size, cfg.SpawnJitter, cfg.Radius, cfg.MaxHp, cfg.Speed,
                              cfg.BulletSpeed, cfg.BulletRadius, cfg.BulletLifetime, cfg.Damage, cfg.Cooldown,
                              cfg.DashMultiplier, cfg.DashDuration, cfg.DashCooldown, cfg.RayRange };
            for (int i = 0; i < python.Length; i++)
            {
                Assert.AreEqual(python[i], ours[i], 1e-12, "setting " + i);
            }
            Assert.AreEqual(c.maxBullets, cfg.MaxBullets);
            Assert.AreEqual(c.rays, cfg.Rays);
            Assert.AreEqual(c.trackedBullets, cfg.TrackedBullets);
            Assert.AreEqual(c.obsDim, cfg.ObsDim);
            Assert.AreEqual(c.maxSteps, cfg.MaxSteps);
            CollectionAssert.AreEqual(c.actionSizes, cfg.ActionSizes);
            CollectionAssert.AreEqual(c.obstacles, cfg.Obstacles);
            CollectionAssert.AreEqual(c.spawns, cfg.Spawns);
            CollectionAssert.AreEqual(c.turnDegrees, cfg.TurnDegrees);
        }

        [TestCase(0)]
        [TestCase(1)]
        [TestCase(2)]
        [TestCase(3)]
        public void RecordedGameReplaysLikePython(int index)
        {
            TraceFile traces = TraceFile.Load();
            Assume.That(index < traces.games.Length, "only " + traces.games.Length + " games recorded");
            TraceGame g = traces.games[index];
            var cfg = new GameConfig();
            var game = new Game(cfg);
            var buffer = new float[2 * cfg.ObsDim];
            int k = cfg.MaxBullets;

            game.ResetTo(g.startPos[0], g.startPos[1], g.startPos[2], g.startPos[3], g.startAngle[0], g.startAngle[1]);
            CheckObservations(game, g.startObs, 0, buffer, g.name + ", before the first step");

            for (int t = 0; t < g.steps; t++)
            {
                string at = g.name + ", step " + (t + 1);
                int a = 8 * t;
                game.Step(new PlayerAction(g.actions[a], g.actions[a + 1], g.actions[a + 2], g.actions[a + 3]),
                          new PlayerAction(g.actions[a + 4], g.actions[a + 5], g.actions[a + 6], g.actions[a + 7]));
                for (int p = 0; p < 2; p++)
                {
                    string who = " of player " + p;
                    Near(game.X[p], g.pos[4 * t + 2 * p], Tolerance, at, "x" + who);
                    Near(game.Y[p], g.pos[4 * t + 2 * p + 1], Tolerance, at, "y" + who);
                    // Angles are compared around the circle: -pi and pi are the same facing.
                    Near(Game.WrapAngle(game.Angle[p] - g.angle[2 * t + p]), 0.0, Tolerance, at, "facing" + who);
                    Near(game.VelX[p], g.vel[4 * t + 2 * p], VelocityTolerance, at, "velocity x" + who);
                    Near(game.VelY[p], g.vel[4 * t + 2 * p + 1], VelocityTolerance, at, "velocity y" + who);
                    Near(game.Hp[p], g.hp[2 * t + p], 0.0, at, "hp" + who);
                    Same(game.ShootCd[p], g.timers[6 * t + p], at, "shoot cooldown" + who);
                    Same(game.DashCd[p], g.timers[6 * t + 2 + p], at, "dash cooldown" + who);
                    Same(game.DashLeft[p], g.timers[6 * t + 4 + p], at, "dash steps left" + who);
                    Same(game.Unseen[p], g.unseen[2 * t + p], at, "steps unseen" + who);
                    Near(game.LastSeenX[p], g.lastSeen[4 * t + 2 * p], Tolerance, at, "last seen x" + who);
                    Near(game.LastSeenY[p], g.lastSeen[4 * t + 2 * p + 1], Tolerance, at, "last seen y" + who);
                }
                for (int b = 0; b < 2 * k; b++)
                {
                    string bullet = "bullet " + (b % k) + " of player " + (b / k);
                    Same(game.BulletAlive[b] ? 1 : 0, g.bullets[6 * t + b], at, bullet + " alive");
                    if (game.BulletAlive[b])
                    {
                        Near(game.BulletX[b], g.bulletPos[12 * t + 2 * b], Tolerance, at, bullet + " x");
                        Near(game.BulletY[b], g.bulletPos[12 * t + 2 * b + 1], Tolerance, at, bullet + " y");
                    }
                }
                Same(game.Visible ? 1 : 0, g.visible[t], at, "visible");
                Same(game.Done ? 1 : 0, g.done[t], at, "done");
                Same(game.Outcome, g.outcome[t], at, "outcome");
                CheckObservations(game, g.obs, 2 * cfg.ObsDim * t, buffer, at);
            }
            Assert.IsTrue(game.Done, g.name + " should be over after its recorded steps");
        }

        [Test]
        public void DecisionGateDecidesOnTheSameStepsAsPython()
        {
            foreach (TraceGate recorded in TraceFile.Load().gates)
            {
                var gate = new DecisionGate(recorded.every);
                for (int s = 0; s < recorded.decide.Length; s++)
                {
                    if (s == recorded.newGameAt)
                    {
                        gate.NewGame();
                    }
                    Same(gate.Step() ? 1 : 0, recorded.decide[s], "every " + recorded.every, "decision at step " + s);
                }
            }
        }

        [Test]
        public void ControlsMatchPython()
        {
            TraceControls c = TraceFile.Load().controls;
            var cfg = new GameConfig();
            for (int i = 0; i < c.wrapIn.Length; i++)
            {
                Near(Game.WrapAngle(c.wrapIn[i]), c.wrapOut[i], 1e-12, "wrap", "wrap of " + c.wrapIn[i]);
            }
            foreach (int k in new[] { -5, -3, -1, 1, 3, 5 })
            {
                Assert.AreEqual(-Math.PI, Game.WrapAngle(k * Math.PI), "wrap of " + k + " pi");  // [-pi, pi)
            }
            Assert.AreEqual(0.0, Game.WrapAngle(0.0));
            for (int i = 0; i < c.turnDegrees.Length; i++)
            {
                double diff = c.turnDegrees[i] * (Math.PI / 180.0);  // as numpy.radians
                Same(Controls.TurnTowards(diff, cfg.TurnRadians), c.turnsForDegrees[i], "turn",
                     "turn for " + c.turnDegrees[i] + " degrees");
            }
            for (int i = 0; i < c.turnDiffs.Length; i++)
            {
                Same(Controls.TurnTowards(c.turnDiffs[i], cfg.TurnRadians), c.turns[i], "turn", "turn for " + c.turnDiffs[i]);
            }
            for (int i = 0; i < c.moveEighths.Length; i++)
            {
                double angle = c.moveEighths[i] * Math.PI / 8;
                Same(Controls.MoveTowards(angle), c.movesForEighths[i], "move", "move for " + c.moveEighths[i] + "/8 turn");
            }
            for (int i = 0; i < c.moveAngles.Length; i++)
            {
                Same(Controls.MoveTowards(c.moveAngles[i]), c.moves[i], "move", "move for " + c.moveAngles[i]);
            }
        }
    }
}
