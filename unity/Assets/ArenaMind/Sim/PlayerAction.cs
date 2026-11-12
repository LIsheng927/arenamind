namespace ArenaMind.Sim
{
    /// <summary>
    /// One player's choices for one step, the same four numbers a policy picks:
    /// move 0-8 (see GameConfig.MoveDirections), turn 0-4 (see
    /// GameConfig.TurnDegrees), shoot 0 or 1, dash 0 or 1.
    /// </summary>
    public readonly struct PlayerAction
    {
        public readonly int Move;
        public readonly int Turn;
        public readonly int Shoot;
        public readonly int Dash;

        public PlayerAction(int move, int turn, int shoot, int dash)
        {
            Move = move;
            Turn = turn;
            Shoot = shoot;
            Dash = dash;
        }

        /// <summary>Standing still, not turning, not shooting.</summary>
        public static PlayerAction Idle()
        {
            return new PlayerAction(0, 2, 0, 0);
        }
    }
}
