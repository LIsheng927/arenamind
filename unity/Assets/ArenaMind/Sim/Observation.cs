using System;

namespace ArenaMind.Sim
{
    /// <summary>
    /// What a player is allowed to know, as the 49 numbers the policy was
    /// trained on: a port of arena/obs.py, block by block and in the same order.
    ///
    ///     self      10  hp, shoot cooldown, dash cooldown, velocity (fwd, right),
    ///                   x, y, cos(facing), sin(facing), time left
    ///     opponent   8  visible, position (fwd, right), distance, hp,
    ///                   velocity (fwd, right), time since last seen
    ///     rays      16  distance to the nearest wall or cover, from straight
    ///                   ahead going clockwise
    ///     bullets   15  the 3 nearest visible enemy bullets: present,
    ///                   position (fwd, right), velocity (fwd, right)
    ///
    /// Positions and velocities are relative to the player, split into
    /// forward and right parts. Player 1 sees the map turned by 180 degrees,
    /// so both players see the same numbers in mirrored situations.
    /// </summary>
    public static class Observation
    {
        public const double UnseenScale = 10.0;  // seconds; "time since last seen" stops at 1 here

        /// <summary>Writes Config.ObsDim numbers for player into output, starting at offset.</summary>
        public static void Write(Game game, int player, float[] output, int offset)
        {
            GameConfig cfg = game.Config;
            int p = player, q = 1 - player;
            double size = cfg.Size;
            double maxSpeed = cfg.Speed * cfg.DashMultiplier;
            double c = Math.Cos(game.Angle[p]), s = Math.Sin(game.Angle[p]);
            double mx = game.X[p], my = game.Y[p];
            int o = offset;

            // --- self ---
            double vf = game.VelX[p] * c + game.VelY[p] * s;
            double vr = -game.VelX[p] * s + game.VelY[p] * c;
            bool flip = p == 1;  // player 1 sees the map turned by 180 degrees
            output[o++] = (float)(game.Hp[p] / cfg.MaxHp);
            output[o++] = (float)((double)game.ShootCd[p] / cfg.ShootCooldownSteps);
            output[o++] = (float)((double)game.DashCd[p] / cfg.DashCooldownSteps);
            output[o++] = (float)(vf / maxSpeed);
            output[o++] = (float)(vr / maxSpeed);
            output[o++] = (float)((flip ? size - mx : mx) / size);
            output[o++] = (float)((flip ? size - my : my) / size);
            output[o++] = (float)(flip ? -c : c);
            output[o++] = (float)(flip ? -s : s);
            output[o++] = (float)((double)(cfg.MaxSteps - game.T) / cfg.MaxSteps);

            // --- opponent: where it was last seen, which is where it is while visible ---
            double relX = game.LastSeenX[p] - mx, relY = game.LastSeenY[p] - my;
            double pf = relX * c + relY * s, pr = -relX * s + relY * c;
            double ovf = game.VelX[q] * c + game.VelY[q] * s, ovr = -game.VelX[q] * s + game.VelY[q] * c;
            double unseenSeconds = game.Unseen[p] * cfg.Dt;
            output[o++] = game.Visible ? 1f : 0f;
            output[o++] = (float)(pf / size);
            output[o++] = (float)(pr / size);
            output[o++] = (float)(Geometry.Length(relX, relY) / size);
            output[o++] = (float)(game.Hp[q] / cfg.MaxHp);  // known even while hidden: you count your hits
            output[o++] = (float)((game.Visible ? ovf : 0.0) / maxSpeed);
            output[o++] = (float)((game.Visible ? ovr : 0.0) / maxSpeed);
            output[o++] = (float)Math.Min(unseenSeconds / UnseenScale, 1.0);

            // --- range sensors ---
            double between = 2 * Math.PI / cfg.Rays;
            for (int i = 0; i < cfg.Rays; i++)
            {
                double a = game.Angle[p] + i * between;
                double dist = Geometry.Raycast(mx, my, Math.Cos(a), Math.Sin(a), cfg.Obstacles, size, cfg.RayRange);
                output[o++] = (float)(dist / cfg.RayRange);
            }

            // --- the nearest enemy bullets this player can see ---
            int k = cfg.MaxBullets;
            bool[] seen = new bool[k];
            double[] distance = new double[k];
            int[] order = new int[k];
            for (int slot = 0; slot < k; slot++)
            {
                int i = q * k + slot;  // the enemy's bullets
                seen[slot] = game.BulletAlive[i]
                             && Geometry.LineOfSight(mx, my, game.BulletX[i], game.BulletY[i], cfg.Obstacles);
                distance[slot] = seen[slot]
                    ? Geometry.Length(game.BulletX[i] - mx, game.BulletY[i] - my)
                    : double.PositiveInfinity;
                order[slot] = slot;
            }
            // Nearest first; equal distances keep their slot order, as a stable argsort does.
            for (int a = 1; a < k; a++)
            {
                int item = order[a];
                int b = a - 1;
                while (b >= 0 && distance[order[b]] > distance[item])
                {
                    order[b + 1] = order[b];
                    b--;
                }
                order[b + 1] = item;
            }
            for (int j = 0; j < cfg.TrackedBullets; j++)
            {
                int slot = order[j];
                if (!seen[slot])
                {
                    for (int z = 0; z < 5; z++)
                    {
                        output[o++] = 0f;  // an empty slot is all zeros
                    }
                    continue;
                }
                int i = q * k + slot;
                double rx = game.BulletX[i] - mx, ry = game.BulletY[i] - my;
                double bvx = game.BulletVx[i] / cfg.Dt, bvy = game.BulletVy[i] / cfg.Dt;  // per second
                output[o++] = 1f;
                output[o++] = (float)((rx * c + ry * s) / size);
                output[o++] = (float)((-rx * s + ry * c) / size);
                output[o++] = (float)((bvx * c + bvy * s) / cfg.BulletSpeed);
                output[o++] = (float)((-bvx * s + bvy * c) / cfg.BulletSpeed);
            }
        }
    }
}
