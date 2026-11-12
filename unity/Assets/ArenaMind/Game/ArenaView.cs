using ArenaMind.Sim;
using UnityEngine;

namespace ArenaMind.Play
{
    /// <summary>
    /// Draws a game with sprites made at start-up, so the project needs no
    /// art assets. The game itself runs 10 steps per second; positions are
    /// blended between the last two steps so it moves smoothly at any frame rate.
    ///
    /// The game's y axis points down the screen and Unity's points up, so a
    /// game point (x, y) is drawn at (x, Size - y) and a facing of a radians
    /// (clockwise on screen) is a rotation of -a about the z axis.
    /// </summary>
    public sealed class ArenaView
    {
        // The colours of the Python viewer (arena/render.py).
        public static readonly Color Background = Rgb(250, 249, 245);
        public static readonly Color Outside = Rgb(232, 230, 223);
        public static readonly Color Grid = Rgb(236, 234, 227);
        public static readonly Color GridMajor = Rgb(222, 220, 212);
        public static readonly Color Cover = Rgb(176, 174, 165);
        public static readonly Color Ink = Rgb(60, 60, 58);
        public static readonly Color HpBack = Rgb(214, 212, 203);
        public static readonly Color[] Players = { Rgb(55, 138, 221), Rgb(216, 90, 48) };  // you, the NPC

        private readonly GameConfig cfg;
        private readonly Material material;
        private readonly Sprite square, circle, ring;
        private readonly Transform root;
        private readonly Transform[] body = new Transform[2];
        private readonly Transform[] hpBack = new Transform[2], hpFill = new Transform[2];
        private readonly SpriteRenderer[] dashGlow = new SpriteRenderer[2];
        private readonly Transform[] bullets;
        private readonly Transform sightLine, memory;
        private readonly SpriteRenderer memoryRenderer;

        // The state at the previous step, to blend from.
        private readonly float[] lastX = new float[2], lastY = new float[2], lastAngle = new float[2];
        private readonly bool[] lastAlive;
        private readonly float[] lastBulletX, lastBulletY;

        public static Color Rgb(int r, int g, int b)
        {
            return new Color(r / 255f, g / 255f, b / 255f);
        }

        public ArenaView(GameConfig config)
        {
            cfg = config;
            // Unlit, so the sprites look the same with or without 2D lights.
            Shader shader = Shader.Find("Universal Render Pipeline/2D/Sprite-Unlit-Default");
            if (shader == null)
            {
                shader = Shader.Find("Sprites/Default");
            }
            if (shader != null)
            {
                material = new Material(shader);
            }
            square = MakeSprite(4, false, 0f);
            circle = MakeSprite(128, true, 0f);
            ring = MakeSprite(128, true, 0.12f);
            root = new GameObject("Arena").transform;

            float size = (float)cfg.Size;
            Place(Make("outside", square, Outside, -3), size / 2, size / 2, size + 60f, size + 60f);
            Place(Make("floor", square, Background, -2), size / 2, size / 2, size, size);
            for (int i = 1; i < (int)cfg.Size; i++)
            {
                Color color = i % 5 == 0 ? GridMajor : Grid;
                Place(Make("grid", square, color, -1), i, size / 2, 0.04f, size);
                Place(Make("grid", square, color, -1), size / 2, i, size, 0.04f);
            }
            for (int r = 0; r + 3 < cfg.Obstacles.Length; r += 4)
            {
                float x0 = (float)cfg.Obstacles[r], y0 = (float)cfg.Obstacles[r + 1];
                float x1 = (float)cfg.Obstacles[r + 2], y1 = (float)cfg.Obstacles[r + 3];
                Place(Make("cover", square, Cover, 0), (x0 + x1) / 2, size - (y0 + y1) / 2, x1 - x0, y1 - y0);
            }

            sightLine = Make("line of sight", square, new Color(Ink.r, Ink.g, Ink.b, 0.18f), 1).transform;
            memoryRenderer = Make("where the NPC thinks you are", ring,
                                  new Color(Players[0].r, Players[0].g, Players[0].b, 0.7f), 2);
            memory = memoryRenderer.transform;
            float diameter = (float)(2 * cfg.Radius);
            memory.localScale = new Vector3(diameter, diameter, 1f);

            for (int p = 0; p < 2; p++)
            {
                dashGlow[p] = Make("dash", circle, new Color(Players[p].r, Players[p].g, Players[p].b, 0.25f), 3);
                body[p] = Make(p == 0 ? "you" : "NPC", circle, Players[p], 4).transform;
                body[p].localScale = new Vector3(diameter, diameter, 1f);
                // The barrel shows where the player faces (local +x).
                Transform barrel = Make("barrel", square, Ink, 5).transform;
                barrel.SetParent(body[p], false);
                barrel.localPosition = new Vector3(0.45f, 0f, 0f);
                barrel.localScale = new Vector3(0.5f, 0.16f, 1f);
                hpBack[p] = Make("hp", square, HpBack, 6).transform;
                hpFill[p] = Make("hp left", square, Players[p], 7).transform;
            }
            int n = 2 * cfg.MaxBullets;
            bullets = new Transform[n];
            lastAlive = new bool[n];
            lastBulletX = new float[n];
            lastBulletY = new float[n];
            for (int b = 0; b < n; b++)
            {
                bullets[b] = Make("bullet", circle, Players[b / cfg.MaxBullets], 8).transform;
                float d = (float)(2 * cfg.BulletRadius) * 1.8f;  // drawn a little larger than it is
                bullets[b].localScale = new Vector3(d, d, 1f);
            }
        }

        /// <summary>Remembers the state before a step, to blend from.</summary>
        public void BeforeStep(Game game)
        {
            for (int p = 0; p < 2; p++)
            {
                lastX[p] = (float)game.X[p];
                lastY[p] = (float)game.Y[p];
                lastAngle[p] = (float)game.Angle[p];
            }
            for (int b = 0; b < bullets.Length; b++)
            {
                lastAlive[b] = game.BulletAlive[b];
                lastBulletX[b] = (float)game.BulletX[b];
                lastBulletY[b] = (float)game.BulletY[b];
            }
        }

        /// <summary>Draws the game blended from the previous step (blend 0) to the current one (blend 1).</summary>
        public void Draw(Game game, float blend, bool showMemory)
        {
            float size = (float)cfg.Size;
            var at = new Vector3[2];
            for (int p = 0; p < 2; p++)
            {
                float x = Mathf.Lerp(lastX[p], (float)game.X[p], blend);
                float y = Mathf.Lerp(lastY[p], (float)game.Y[p], blend);
                float turn = (float)Game.WrapAngle(game.Angle[p] - lastAngle[p]);
                float angle = lastAngle[p] + turn * blend;
                at[p] = new Vector3(x, size - y, 0f);
                body[p].position = at[p];
                body[p].rotation = Quaternion.Euler(0f, 0f, -angle * Mathf.Rad2Deg);

                bool dashing = game.DashLeft[p] > 0;
                dashGlow[p].enabled = dashing;
                dashGlow[p].transform.position = at[p];
                float glow = (float)(2 * cfg.Radius) * 1.7f;
                dashGlow[p].transform.localScale = new Vector3(glow, glow, 1f);

                float width = 1.2f, left = (float)(game.Hp[p] / cfg.MaxHp) * width;
                hpBack[p].position = at[p] + new Vector3(0f, 0.85f, 0f);
                hpBack[p].localScale = new Vector3(width, 0.14f, 1f);
                hpFill[p].position = at[p] + new Vector3((left - width) / 2, 0.85f, 0f);
                hpFill[p].localScale = new Vector3(left, 0.14f, 1f);
            }

            // A faint line while the two can see each other.
            sightLine.gameObject.SetActive(game.Visible);
            if (game.Visible)
            {
                Vector3 d = at[1] - at[0];
                sightLine.position = (at[0] + at[1]) / 2;
                sightLine.rotation = Quaternion.Euler(0f, 0f, Mathf.Atan2(d.y, d.x) * Mathf.Rad2Deg);
                sightLine.localScale = new Vector3(d.magnitude, 0.05f, 1f);
            }

            // Out of sight, the NPC only knows where it last saw you: this ring.
            bool remembering = showMemory && !game.Visible;
            memoryRenderer.enabled = remembering;
            if (remembering)
            {
                memory.position = new Vector3((float)game.LastSeenX[1], size - (float)game.LastSeenY[1], 0f);
            }

            for (int b = 0; b < bullets.Length; b++)
            {
                bool alive = game.BulletAlive[b];
                bullets[b].gameObject.SetActive(alive);
                if (alive)
                {
                    float x = (float)game.BulletX[b], y = (float)game.BulletY[b];
                    if (lastAlive[b])
                    {
                        x = Mathf.Lerp(lastBulletX[b], x, blend);
                        y = Mathf.Lerp(lastBulletY[b], y, blend);
                    }
                    bullets[b].position = new Vector3(x, size - y, 0f);
                }
            }
        }

        /// <summary>Game coordinates of a point on the screen (pixels, origin bottom-left).</summary>
        public Vector2 ScreenToGame(Camera camera, Vector2 screen)
        {
            Vector3 world = camera.ScreenToWorldPoint(new Vector3(screen.x, screen.y, -camera.transform.position.z));
            return new Vector2(world.x, (float)cfg.Size - world.y);
        }

        private SpriteRenderer Make(string name, Sprite sprite, Color color, int order)
        {
            var go = new GameObject(name);
            go.transform.SetParent(root, false);
            var renderer = go.AddComponent<SpriteRenderer>();
            renderer.sprite = sprite;
            renderer.color = color;
            renderer.sortingOrder = order;
            if (material != null)
            {
                renderer.sharedMaterial = material;
            }
            return renderer;
        }

        private static void Place(SpriteRenderer renderer, float x, float y, float width, float height)
        {
            renderer.transform.position = new Vector3(x, y, 0f);
            renderer.transform.localScale = new Vector3(width, height, 1f);
        }

        // A white sprite one unit across: a square, a disc, or a ring of the given thickness (in units).
        private static Sprite MakeSprite(int pixels, bool round, float ring)
        {
            var texture = new Texture2D(pixels, pixels, TextureFormat.RGBA32, false);
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.filterMode = FilterMode.Bilinear;
            var colors = new Color32[pixels * pixels];
            float c = (pixels - 1) / 2f, outer = pixels / 2f, inner = outer - ring * pixels;
            for (int y = 0; y < pixels; y++)
            {
                for (int x = 0; x < pixels; x++)
                {
                    float alpha = 1f;
                    if (round)
                    {
                        float d = Mathf.Sqrt((x - c) * (x - c) + (y - c) * (y - c));
                        alpha = Mathf.Clamp01(outer - d);  // one pixel of soft edge
                        if (ring > 0)
                        {
                            alpha = Mathf.Min(alpha, Mathf.Clamp01(d - inner));
                        }
                    }
                    colors[y * pixels + x] = new Color32(255, 255, 255, (byte)Mathf.RoundToInt(alpha * 255));
                }
            }
            texture.SetPixels32(colors);
            texture.Apply();
            return Sprite.Create(texture, new Rect(0, 0, pixels, pixels), new Vector2(0.5f, 0.5f), pixels);
        }
    }
}
