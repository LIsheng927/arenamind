using System;

namespace ArenaMind.Sim
{
    /// <summary>
    /// Turning what a human wants (aim here, go that way on screen) into the
    /// same action choices the policy has, like tools/play.py with
    /// turn_towards and move_towards of arena/bots.py. Human games are then
    /// played under the same rules as the policy's, and can be recorded as
    /// demonstrations later.
    /// </summary>
    public static class Controls
    {
        /// <summary>The turn choice that brings the facing closest to diff radians away (the first on a tie).</summary>
        public static int TurnTowards(double diff, double[] turnRadians)
        {
            int best = 0;
            double bestError = Math.Abs(diff - turnRadians[0]);
            for (int i = 1; i < turnRadians.Length; i++)
            {
                double error = Math.Abs(diff - turnRadians[i]);
                if (error < bestError)
                {
                    best = i;
                    bestError = error;
                }
            }
            return best;
        }

        /// <summary>
        /// The move whose direction is closest to relAngle (0 straight ahead,
        /// clockwise): one of the 8 directions 1-8. Halfway between two, the
        /// even sector wins, as numpy's round does.
        /// </summary>
        public static int MoveTowards(double relAngle)
        {
            long sector = (long)Math.Round(relAngle / (Math.PI / 4));  // rounds half to even
            int s = (int)(((sector % 8) + 8) % 8);  // 0 ahead, 2 right, 4 back, 6 left
            return 1 + s;
        }

        /// <summary>
        /// A human's action: turn towards the point (aimX, aimY) as far as one
        /// step allows, move towards the screen direction (dx, dy) (x right,
        /// y down; both 0 to stand still), and shoot or dash if asked.
        /// </summary>
        public static PlayerAction Human(Game game, int player, double aimX, double aimY,
                                         double dx, double dy, bool shoot, bool dash)
        {
            GameConfig cfg = game.Config;
            double angle = game.Angle[player];
            double aim = Math.Atan2(aimY - game.Y[player], aimX - game.X[player]);
            int turn = TurnTowards(Game.WrapAngle(aim - angle), cfg.TurnRadians);
            double newAngle = angle + cfg.TurnRadians[turn];  // moves happen after the turn
            int move = 0;
            if (dx != 0 || dy != 0)
            {
                move = MoveTowards(Game.WrapAngle(Math.Atan2(dy, dx) - newAngle));
            }
            return new PlayerAction(move, turn, shoot ? 1 : 0, dash ? 1 : 0);
        }
    }
}
