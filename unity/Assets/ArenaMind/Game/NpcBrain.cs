using System;
using ArenaMind.Sim;

namespace ArenaMind.Play
{
    /// <summary>
    /// One NPC player: what it sees, when it decides and what it does, as
    /// DifficultyBot in rl/difficulty.py plays a level of the ladder. On the
    /// steps its DecisionGate allows, it writes its 49 observation numbers and
    /// its personality sliders into the network's input, gets the 18 scores
    /// from Sentis and draws one choice per head; on the other steps it
    /// repeats its last action.
    /// </summary>
    public sealed class NpcBrain
    {
        public readonly int Player;
        public readonly DecisionGate Gate;
        public float[] Sliders;  // personality: extra inputs of the network
        public int Decisions;  // in the current game
        public float[] LastScores;  // of the last decision, null before the first
        public double LastMilliseconds;  // Sentis time of the last decision

        private readonly ArenaPolicy policy;
        private readonly GameConfig cfg;
        private readonly Random random;
        private readonly float[] input;
        private PlayerAction action = PlayerAction.Idle();

        public NpcBrain(ArenaPolicy policy, GameConfig cfg, int player, int inputs, double every, float[] sliders,
                        Random random)
        {
            if (inputs != cfg.ObsDim + sliders.Length)
            {
                throw new ArgumentException("the network takes " + inputs + " inputs, but " + cfg.ObsDim
                                            + " observation numbers and " + sliders.Length + " sliders were given");
            }
            this.policy = policy;
            this.cfg = cfg;
            this.random = random;
            Player = player;
            Sliders = sliders;
            Gate = new DecisionGate(every);
            input = new float[inputs];
        }

        /// <summary>This step's action. Call once per step, before Game.Step.</summary>
        public PlayerAction Act(Game game)
        {
            if (Gate.Step())
            {
                Observation.Write(game, Player, input, 0);
                for (int i = 0; i < Sliders.Length; i++)
                {
                    input[cfg.ObsDim + i] = Sliders[i];
                }
                var watch = System.Diagnostics.Stopwatch.StartNew();
                LastScores = policy.Scores(input);
                LastMilliseconds = watch.Elapsed.TotalMilliseconds;
                action = ActionSampler.Sample(LastScores, cfg.ActionSizes, 1.0, random);
                Decisions++;
            }
            return action;
        }

        /// <summary>Call when a new game starts: the first step is always a decision.</summary>
        public void NewGame()
        {
            Gate.NewGame();
            action = PlayerAction.Idle();
            Decisions = 0;
            LastScores = null;
        }
    }
}
