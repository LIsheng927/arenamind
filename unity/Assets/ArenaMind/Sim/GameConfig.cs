using System;

namespace ArenaMind.Sim
{
    /// <summary>
    /// Every number of the game, the same as arena/config.py in the Python
    /// project. The tests compare them with the values recorded by
    /// tools/export_traces.py, so a change on one side only is caught.
    ///
    /// Coordinates are those of the Python game: x grows to the right, y grows
    /// downward, the arena is [0, Size] x [0, Size]. Angle 0 faces +x and a
    /// positive angle turns clockwise on screen.
    /// </summary>
    public sealed class GameConfig
    {
        public readonly double Dt = 0.1;  // seconds per step: 10 decisions per second
        public readonly double MaxTime = 60.0;  // then the player with more HP wins
        public readonly double Size = 20.0;

        // (x min, y min, x max, y max) of every obstacle, one after another
        public readonly double[] Obstacles = {
            9.0, 9.0, 11.0, 11.0,
            4.0, 11.0, 8.0, 12.0,
            12.0, 8.0, 16.0, 9.0,
            6.0, 6.0, 7.0, 7.0,
            13.0, 13.0, 14.0, 14.0,
        };

        public readonly double[] Spawns = { 3.0, 17.0, 17.0, 3.0 };  // x0, y0, x1, y1
        public readonly double SpawnJitter = 0.5;

        public readonly double Radius = 0.5;
        public readonly double MaxHp = 100.0;
        public readonly double Speed = 4.0;  // units per second
        public readonly double[] TurnDegrees = { -30.0, -8.0, 0.0, 8.0, 30.0 };  // per step

        public readonly double BulletSpeed = 15.0;
        public readonly double BulletRadius = 0.1;
        public readonly double BulletLifetime = 1.0;
        public readonly double Damage = 20.0;
        public readonly double Cooldown = 0.4;
        public readonly int MaxBullets = 3;  // bullet slots per player

        public readonly double DashMultiplier = 3.0;
        public readonly double DashDuration = 0.3;
        public readonly double DashCooldown = 3.0;

        public readonly int Rays = 16;
        public readonly double RayRange = 10.0;
        public readonly int TrackedBullets = 3;

        // The 9 movement choices as (forward, right) in the player's own frame.
        // Diagonals have length 1 too, so moving diagonally is not faster.
        private static readonly double D = Math.Sqrt(0.5);
        public static readonly double[] MoveDirections = {
            0.0, 0.0,  // 0 stay
            1.0, 0.0,  // 1 forward
            D, D,  // 2 forward-right
            0.0, 1.0,  // 3 right
            -D, D,  // 4 back-right
            -1.0, 0.0,  // 5 back
            -D, -D,  // 6 back-left
            0.0, -1.0,  // 7 left
            D, -D,  // 8 forward-left
        };

        // Worked out from the numbers above, as the Python properties do.
        public readonly int MaxSteps, ShootCooldownSteps, BulletLifetimeSteps, DashSteps, DashCooldownSteps;
        public readonly int DashSubsteps;  // a dashing player makes this many normal moves per step
        public readonly double MovePerStep, BulletPerStep;
        public readonly double[] TurnRadians;
        public readonly int ObsDim;  // 10 self + 8 opponent + rays + 5 per tracked bullet
        public readonly int[] ActionSizes;  // choices for move, turn, shoot, dash

        public GameConfig()
        {
            MaxSteps = Steps(MaxTime);
            ShootCooldownSteps = Steps(Cooldown);
            BulletLifetimeSteps = Steps(BulletLifetime);
            DashSteps = Steps(DashDuration);
            DashCooldownSteps = Steps(DashCooldown);
            DashSubsteps = (int)Math.Round(DashMultiplier);
            MovePerStep = Speed * Dt;
            BulletPerStep = BulletSpeed * Dt;
            TurnRadians = new double[TurnDegrees.Length];
            for (int i = 0; i < TurnDegrees.Length; i++)
            {
                TurnRadians[i] = TurnDegrees[i] * (Math.PI / 180.0);  // as numpy.radians
            }
            ObsDim = 10 + 8 + Rays + TrackedBullets * 5;
            ActionSizes = new int[] { MoveDirections.Length / 2, TurnDegrees.Length, 2, 2 };
        }

        // A duration as a whole number of steps, or an error.
        private int Steps(double seconds)
        {
            int steps = (int)Math.Round(seconds / Dt);
            if (Math.Abs(steps * Dt - seconds) > 1e-9)
            {
                throw new ArgumentException(seconds + "s is not a whole number of " + Dt + "s steps");
            }
            return steps;
        }
    }
}
