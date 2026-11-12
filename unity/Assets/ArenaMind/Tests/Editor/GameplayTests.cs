using System;
using ArenaMind.Sim;
using NUnit.Framework;

namespace ArenaMind.Tests
{
    /// <summary>The parts of the C# game that the recorded games do not cover.</summary>
    public class GameplayTests
    {
        [Test]
        public void SamplingFollowsTheSoftmaxOfTheScores()
        {
            float[] scores = { 1f, 0f, -1f, 2f, 2f };  // a head of 3 choices, then one of 2
            var probabilities = new double[3];
            ActionSampler.Probabilities(scores, 0, 3, probabilities);
            Assert.AreEqual(Math.E / (Math.E + 1 + 1 / Math.E), probabilities[0], 1e-12);

            var random = new Random(1);
            var counts = new int[3];
            const int draws = 20000;
            for (int n = 0; n < draws; n++)
            {
                counts[ActionSampler.Pick(scores, 0, 3, 1.0, random.NextDouble())] += 1;
            }
            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(probabilities[i], (double)counts[i] / draws, 0.015, "choice " + i);
            }
            Assert.AreEqual(0, ActionSampler.Pick(scores, 0, 3, 0.0, 0.99), "temperature 0 takes the largest");
            Assert.AreEqual(0, ActionSampler.Pick(scores, 3, 2, 0.0, 0.99), "a tie goes to the first");
        }

        [Test]
        public void SampleGivesOneChoicePerHead()
        {
            var cfg = new GameConfig();
            var scores = new float[18];
            scores[4] = 50f;  // move 4
            scores[9 + 1] = 50f;  // turn 1
            scores[14 + 1] = 50f;  // shoot
            scores[16] = 50f;  // no dash
            PlayerAction a = ActionSampler.Sample(scores, cfg.ActionSizes, 1.0, new Random(0));
            Assert.AreEqual(new[] { 4, 1, 1, 0 }, new[] { a.Move, a.Turn, a.Shoot, a.Dash });
        }

        [Test]
        public void HumanControlsUseScreenDirections()
        {
            var game = new Game(new GameConfig());
            game.ResetTo(5, 15, 15, 5, 0.0, 0.0);  // player 0 faces +x (right on screen)
            PlayerAction a = Controls.Human(game, 0, 9, 15, 1, 0, true, false);  // aim and go right
            Assert.AreEqual(2, a.Turn, "aiming straight ahead needs no turn");
            Assert.AreEqual(1, a.Move, "right on screen is forward");
            Assert.AreEqual(1, a.Shoot);
            a = Controls.Human(game, 0, 9, 15, 0, 1, false, true);  // go down the screen
            Assert.AreEqual(3, a.Move, "down on screen is the player's right when facing right");
            Assert.AreEqual(1, a.Dash);
            a = Controls.Human(game, 0, 5, 20, 0, 0, false, false);  // aim straight down: 90 degrees away
            Assert.AreEqual(4, a.Turn, "the largest clockwise turn");
            Assert.AreEqual(0, a.Move, "no key: stand still");
        }

        [Test]
        public void MirroredPlayersSeeTheSameNumbers()
        {
            var cfg = new GameConfig();
            var game = new Game(cfg);
            // Player 1 stands where player 0 would be with the map turned by 180 degrees.
            game.ResetTo(3.2, 15.7, cfg.Size - 3.2, cfg.Size - 15.7, 0.3, 0.3 - Math.PI);
            var obs = new float[2 * cfg.ObsDim];
            Observation.Write(game, 0, obs, 0);
            Observation.Write(game, 1, obs, cfg.ObsDim);
            for (int k = 0; k < cfg.ObsDim; k++)
            {
                Assert.AreEqual(obs[k], obs[cfg.ObsDim + k], 1e-6, "number " + k);
            }
        }

        [Test]
        public void AFinishedGameMustBeResetBeforeTheNextStep()
        {
            var game = new Game(new GameConfig());
            game.Hp[1] = 0.0;
            game.Step(PlayerAction.Idle(), PlayerAction.Idle());
            Assert.IsTrue(game.Done);
            Assert.AreEqual(1, game.Outcome, "player 0 won");
            Assert.Throws<InvalidOperationException>(() => game.Step(PlayerAction.Idle(), PlayerAction.Idle()));
            game.Reset(new Random(3));
            Assert.IsFalse(game.Done);
            Assert.AreEqual(game.Config.MaxHp, game.Hp[1]);
            Assert.AreEqual(3.0, game.X[0], game.Config.SpawnJitter);
            Assert.AreEqual(17.0, game.Y[0], game.Config.SpawnJitter);
        }

        [Test]
        public void ActionsOutOfRangeAreRefused()
        {
            var game = new Game(new GameConfig());
            Assert.Throws<ArgumentException>(() => game.Step(new PlayerAction(9, 2, 0, 0), PlayerAction.Idle()));
            Assert.Throws<ArgumentException>(() => game.Step(PlayerAction.Idle(), new PlayerAction(0, 5, 0, 0)));
        }
    }
}
