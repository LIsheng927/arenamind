using System;

namespace ArenaMind.Sim
{
    /// <summary>
    /// The difficulty knob of the ladder: the policy decides only every
    /// Every steps and repeats its last action in between, like Handicap in
    /// rl/difficulty.py. Fractions mix the gaps: every step adds one to a
    /// credit and a decision spends Every of it, so 1.5 decides after 2
    /// steps, then 1, then 2, and so on. A new game starts with a decision.
    /// </summary>
    public sealed class DecisionGate
    {
        public double Every;
        private double credit;
        private bool fresh = true;

        public DecisionGate(double every)
        {
            if (every < 1)
            {
                throw new ArgumentException("every must be at least 1, got " + every);
            }
            Every = every;
        }

        /// <summary>Call once per step: whether the policy picks a new action this step.</summary>
        public bool Step()
        {
            bool decide = fresh || credit >= Every - 1e-9;
            credit = (fresh ? 0.0 : credit - (decide ? Every : 0.0)) + 1.0;
            fresh = false;
            return decide;
        }

        /// <summary>Call when a new game starts.</summary>
        public void NewGame()
        {
            fresh = true;
        }
    }

    /// <summary>
    /// Picks an action from the policy's 18 scores: one softmax per head
    /// (move 9, turn 5, shoot 2, dash 2), each drawn on its own, as the
    /// policy was trained and rated.
    /// </summary>
    public static class ActionSampler
    {
        /// <summary>Temperature 1 is the policy as trained; 0 always takes the most likely choice.</summary>
        public static PlayerAction Sample(float[] scores, int[] sizes, double temperature, Random random)
        {
            int start = 0;
            int[] picks = new int[4];
            for (int h = 0; h < 4; h++)
            {
                picks[h] = Pick(scores, start, sizes[h], temperature, random.NextDouble());
                start += sizes[h];
            }
            return new PlayerAction(picks[0], picks[1], picks[2], picks[3]);
        }

        /// <summary>One head: u is a uniform random number in [0, 1).</summary>
        public static int Pick(float[] scores, int start, int count, double temperature, double u)
        {
            int best = 0;
            for (int i = 1; i < count; i++)
            {
                if (scores[start + i] > scores[start + best])
                {
                    best = i;
                }
            }
            if (temperature <= 0)
            {
                return best;
            }
            double[] weight = new double[count];
            double total = 0.0;
            for (int i = 0; i < count; i++)
            {
                // Subtracting the largest score keeps exp() from overflowing.
                weight[i] = Math.Exp((scores[start + i] - scores[start + best]) / temperature);
                total += weight[i];
            }
            double target = u * total;
            double sum = 0.0;
            for (int i = 0; i < count; i++)
            {
                sum += weight[i];
                if (target < sum)
                {
                    return i;
                }
            }
            return count - 1;
        }

        /// <summary>The probability of each choice of one head, for showing what the policy is thinking.</summary>
        public static void Probabilities(float[] scores, int start, int count, double[] output)
        {
            double top = scores[start];
            for (int i = 1; i < count; i++)
            {
                top = Math.Max(top, scores[start + i]);
            }
            double total = 0.0;
            for (int i = 0; i < count; i++)
            {
                output[i] = Math.Exp(scores[start + i] - top);
                total += output[i];
            }
            for (int i = 0; i < count; i++)
            {
                output[i] /= total;
            }
        }
    }
}
