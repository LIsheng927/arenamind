using System;

namespace ArenaMind.Sim
{
    /// <summary>
    /// One game of the arena, a port of arena/env.py (which runs many games at
    /// once; this runs one). Step() advances the game by one tick of
    /// Config.Dt seconds in the same order as the Python version:
    ///
    ///     1. check both players' actions
    ///     2. turn
    ///     3. move, then push players out of walls, cover and each other
    ///     4. advance bullets that are already flying (hits and damage)
    ///     5. fire new bullets
    ///     6. tick timers
    ///     7. check whether the game is over
    ///     8. (rewards: only needed for training, left out here)
    ///     9. update what each player has seen of the other
    ///
    /// Player 0 starts bottom-left, player 1 top-right. Arrays indexed by
    /// player hold player 0 first; bullet arrays are indexed by
    /// owner * MaxBullets + slot.
    /// </summary>
    public sealed class Game
    {
        public readonly GameConfig Config;

        public int T;  // steps since the game started
        public readonly double[] X = new double[2], Y = new double[2];  // player centres
        public readonly double[] Angle = new double[2];  // facing, radians in [-pi, pi)
        public readonly double[] VelX = new double[2], VelY = new double[2];  // movement last step, per second
        public readonly double[] Hp = new double[2];
        public readonly int[] ShootCd = new int[2];  // steps until the next shot is allowed
        public readonly int[] DashCd = new int[2];  // steps until the next dash is allowed
        public readonly int[] DashLeft = new int[2];  // steps of the current dash still to run

        public readonly bool[] BulletAlive;
        public readonly double[] BulletX, BulletY;
        public readonly double[] BulletVx, BulletVy;  // displacement per step
        public readonly int[] BulletLife;  // steps of flight left

        // What each player knows about the other: the opponent is only shown
        // where it was last seen, so that position is remembered here.
        public bool Visible;  // line of sight between the two players
        public readonly double[] LastSeenX = new double[2], LastSeenY = new double[2];
        public readonly int[] Unseen = new int[2];  // steps since this player last saw the other

        public bool Done;
        public int Outcome;  // once Done, for player 0: 1 won, -1 lost, 0 draw

        private readonly double[] point = new double[2];
        private readonly double[] startX = new double[2], startY = new double[2];
        private readonly double[] stepX = new double[2], stepY = new double[2];
        private readonly int[] moves = new int[2];
        private readonly int[] hits = new int[2];

        public Game(GameConfig config)
        {
            Config = config;
            int n = 2 * config.MaxBullets;
            BulletAlive = new bool[n];
            BulletX = new double[n];
            BulletY = new double[n];
            BulletVx = new double[n];
            BulletVy = new double[n];
            BulletLife = new int[n];
            ResetTo(config.Spawns[0], config.Spawns[1], config.Spawns[2], config.Spawns[3], 0.0, 0.0);
        }

        /// <summary>Maps an angle to [-pi, pi), exactly as (a + pi) % (2 pi) - pi does in numpy.</summary>
        public static double WrapAngle(double a)
        {
            double twoPi = 2 * Math.PI;
            double m = (a + Math.PI) % twoPi;  // the sign of a + pi, like C's fmod
            if (m < 0)
            {
                m += twoPi;  // numpy's % takes the sign of the divisor
            }
            else if (m == 0)
            {
                m = 0.0;  // never -0
            }
            return m - Math.PI;
        }

        /// <summary>A new game with the usual random start: near the spawn points, facing anywhere.</summary>
        public void Reset(Random random)
        {
            double j = Config.SpawnJitter;
            double x0 = Config.Spawns[0] + (2 * random.NextDouble() - 1) * j;
            double y0 = Config.Spawns[1] + (2 * random.NextDouble() - 1) * j;
            double x1 = Config.Spawns[2] + (2 * random.NextDouble() - 1) * j;
            double y1 = Config.Spawns[3] + (2 * random.NextDouble() - 1) * j;
            double a0 = (2 * random.NextDouble() - 1) * Math.PI;
            double a1 = (2 * random.NextDouble() - 1) * Math.PI;
            ResetTo(x0, y0, x1, y1, a0, a1);
        }

        /// <summary>A new game with the players placed by hand (the tests start recorded games this way).</summary>
        public void ResetTo(double x0, double y0, double x1, double y1, double angle0, double angle1)
        {
            X[0] = x0;
            Y[0] = y0;
            X[1] = x1;
            Y[1] = y1;
            Angle[0] = angle0;
            Angle[1] = angle1;
            for (int p = 0; p < 2; p++)
            {
                VelX[p] = 0.0;
                VelY[p] = 0.0;
                Hp[p] = Config.MaxHp;
                ShootCd[p] = 0;
                DashCd[p] = 0;
                DashLeft[p] = 0;
                // Spawn points are part of the map: each player starts out
                // knowing where the other began.
                LastSeenX[p] = Config.Spawns[2 * (1 - p)];
                LastSeenY[p] = Config.Spawns[2 * (1 - p) + 1];
                Unseen[p] = 0;
            }
            for (int i = 0; i < BulletAlive.Length; i++)
            {
                BulletAlive[i] = false;
                BulletX[i] = 0.0;
                BulletY[i] = 0.0;
                BulletVx[i] = 0.0;
                BulletVy[i] = 0.0;
                BulletLife[i] = 0;
            }
            T = 0;
            Done = false;
            Outcome = 0;
            UpdateSight(false);
        }

        /// <summary>Advances the game by one tick. Throws once the game is over.</summary>
        public void Step(PlayerAction a0, PlayerAction a1)
        {
            if (Done)
            {
                throw new InvalidOperationException("the game is over; call Reset first");
            }
            CheckAction(a0);
            CheckAction(a1);

            // 2. Turn.
            Angle[0] = WrapAngle(Angle[0] + Config.TurnRadians[a0.Turn]);
            Angle[1] = WrapAngle(Angle[1] + Config.TurnRadians[a1.Turn]);

            // 3. Move.
            MovePlayers(a0, a1);

            // 4. Bullets already in flight.
            AdvanceBullets();

            // 5. New bullets.
            Fire(0, a0.Shoot == 1);
            Fire(1, a1.Shoot == 1);

            // 6. Timers.
            for (int p = 0; p < 2; p++)
            {
                ShootCd[p] = Math.Max(ShootCd[p] - 1, 0);
                DashCd[p] = Math.Max(DashCd[p] - 1, 0);
                DashLeft[p] = Math.Max(DashLeft[p] - 1, 0);
            }
            T += 1;

            // 7. A player at 0 HP loses; at the time limit, more HP wins. Equal HP
            //    (including both players dying in the same step) is a draw.
            Done = Hp[0] <= 0 || Hp[1] <= 0 || T >= Config.MaxSteps;
            Outcome = !Done ? 0 : (Hp[0] > Hp[1] ? 1 : (Hp[0] < Hp[1] ? -1 : 0));

            // 9. What each player can see now.
            UpdateSight(true);
        }

        private void CheckAction(PlayerAction a)
        {
            int[] sizes = Config.ActionSizes;
            if (a.Move < 0 || a.Move >= sizes[0] || a.Turn < 0 || a.Turn >= sizes[1]
                || a.Shoot < 0 || a.Shoot >= sizes[2] || a.Dash < 0 || a.Dash >= sizes[3])
            {
                throw new ArgumentException("action out of range: move " + a.Move + ", turn " + a.Turn
                                            + ", shoot " + a.Shoot + ", dash " + a.Dash);
            }
        }

        private void MovePlayers(PlayerAction a0, PlayerAction a1)
        {
            GameConfig cfg = Config;
            for (int p = 0; p < 2; p++)
            {
                PlayerAction a = p == 0 ? a0 : a1;
                // A dash starts only when both its cooldown and any current dash are over.
                if (a.Dash == 1 && DashCd[p] == 0 && DashLeft[p] == 0)
                {
                    DashLeft[p] = cfg.DashSteps;
                    DashCd[p] = cfg.DashCooldownSteps;
                }
                moves[p] = DashLeft[p] > 0 ? cfg.DashSubsteps : 1;  // normal-length moves this step

                // The chosen (forward, right) direction turned into a world direction.
                double fwd = GameConfig.MoveDirections[2 * a.Move];
                double right = GameConfig.MoveDirections[2 * a.Move + 1];
                double c = Math.Cos(Angle[p]), s = Math.Sin(Angle[p]);
                stepX[p] = (fwd * c - right * s) * cfg.MovePerStep;
                stepY[p] = (fwd * s + right * c) * cfg.MovePerStep;
                startX[p] = X[p];
                startY[p] = Y[p];
            }
            // A dash is split into normal-length moves with collisions resolved
            // after each one, so a player never ends up inside a wall.
            int most = Math.Max(moves[0], moves[1]);
            for (int i = 0; i < most; i++)
            {
                for (int p = 0; p < 2; p++)
                {
                    if (i < moves[p])
                    {
                        X[p] += stepX[p];
                        Y[p] += stepY[p];
                    }
                }
                ResolveCollisions();
            }
            for (int p = 0; p < 2; p++)
            {
                VelX[p] = (X[p] - startX[p]) / cfg.Dt;
                VelY[p] = (Y[p] - startY[p]) / cfg.Dt;
            }
        }

        private void ResolveCollisions()
        {
            double r = Config.Radius;
            // Players may not overlap: push both apart by half the overlap each.
            double dx = X[1] - X[0], dy = Y[1] - Y[0];
            double dist = Geometry.Length(dx, dy);
            bool apart = dist > 1e-9;
            double nx = apart ? dx / dist : 1.0;
            double ny = apart ? dy / dist : 0.0;
            double k = Math.Max(2 * r - dist, 0.0) / 2;
            double px = k * nx, py = k * ny;
            X[0] -= px;
            Y[0] -= py;
            X[1] += px;
            Y[1] += py;
            // Walls and cover always win over the player-player push.
            for (int p = 0; p < 2; p++)
            {
                point[0] = X[p];
                point[1] = Y[p];
                Geometry.PushOutOfRects(point, r, Config.Obstacles);
                X[p] = Geometry.ClampToArena(point[0], r, Config.Size);
                Y[p] = Geometry.ClampToArena(point[1], r, Config.Size);
            }
        }

        private void AdvanceBullets()
        {
            GameConfig cfg = Config;
            double stepLen = cfg.BulletPerStep;
            double reach = cfg.Radius + cfg.BulletRadius;
            hits[0] = 0;
            hits[1] = 0;
            for (int owner = 0; owner < 2; owner++)
            {
                int target = 1 - owner;  // bullets of player p fly at player 1 - p
                for (int slot = 0; slot < cfg.MaxBullets; slot++)
                {
                    int i = owner * cfg.MaxBullets + slot;
                    if (!BulletAlive[i])
                    {
                        continue;
                    }
                    double x0 = BulletX[i], y0 = BulletY[i];
                    double x1 = x0 + BulletVx[i], y1 = y0 + BulletVy[i];
                    // How far along this step's path the bullet first meets cover,
                    // a wall or the opponent's body.
                    double cover = Geometry.SegmentRectsEntry(x0, y0, x1, y1, cfg.Obstacles);
                    double wall = Geometry.RayArenaExit(x0, y0, BulletVx[i] / stepLen, BulletVy[i] / stepLen, cfg.Size);
                    double blocked = Math.Min(cover, wall);
                    double body = Geometry.SegmentCircleEntry(x0, y0, x1, y1, X[target], Y[target], reach);
                    bool hit = body <= stepLen && body <= blocked;
                    bool stopped = !hit && blocked <= stepLen;
                    if (hit)
                    {
                        hits[owner] += 1;
                    }
                    if (hit || stopped)
                    {
                        BulletAlive[i] = false;
                    }
                    else
                    {
                        BulletX[i] = x1;
                        BulletY[i] = y1;
                        BulletLife[i] -= 1;
                        BulletAlive[i] = BulletLife[i] > 0;
                    }
                }
            }
            for (int p = 0; p < 2; p++)
            {
                Hp[p] = Math.Max(Hp[p] - cfg.Damage * hits[1 - p], 0.0);
            }
        }

        private void Fire(int p, bool want)
        {
            GameConfig cfg = Config;
            int free = -1;  // first free bullet slot
            for (int slot = cfg.MaxBullets - 1; slot >= 0; slot--)
            {
                if (!BulletAlive[p * cfg.MaxBullets + slot])
                {
                    free = slot;
                }
            }
            if (!want || ShootCd[p] != 0 || free < 0)
            {
                return;
            }
            // The bullet appears just outside the shooter's body and starts
            // flying on the next step, so the opponent gets a step to react.
            double fx = Math.Cos(Angle[p]), fy = Math.Sin(Angle[p]);
            double reach = cfg.Radius + cfg.BulletRadius;
            double mx = X[p] + fx * reach, my = Y[p] + fy * reach;
            bool inArena = mx >= 0 && mx <= cfg.Size && my >= 0 && my <= cfg.Size;
            ShootCd[p] = cfg.ShootCooldownSteps;
            // Firing point-blank into a wall uses up the shot but makes no bullet.
            if (inArena && !Geometry.InRects(mx, my, cfg.Obstacles))
            {
                int i = p * cfg.MaxBullets + free;
                BulletAlive[i] = true;
                BulletX[i] = mx;
                BulletY[i] = my;
                BulletVx[i] = fx * cfg.BulletPerStep;
                BulletVy[i] = fy * cfg.BulletPerStep;
                BulletLife[i] = cfg.BulletLifetimeSteps;
            }
        }

        private void UpdateSight(bool tick)
        {
            bool vis = Geometry.LineOfSight(X[0], Y[0], X[1], Y[1], Config.Obstacles);
            Visible = vis;
            for (int p = 0; p < 2; p++)
            {
                if (vis)
                {
                    LastSeenX[p] = X[1 - p];
                    LastSeenY[p] = Y[1 - p];
                    Unseen[p] = 0;
                }
                else if (tick)
                {
                    Unseen[p] += 1;
                }
            }
        }
    }
}
