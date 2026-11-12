using System;
using System.IO;
using ArenaMind.Sim;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace ArenaMind.Play
{
    /// <summary>
    /// You (blue, player 0) against the trained NPC (orange, player 1).
    ///
    /// The game runs at 10 steps per second, like the Python version. Before
    /// every step the keyboard and mouse are turned into an action with the
    /// same choices the policy has (Controls.Human), and the NPC gets its 49
    /// observation numbers plus its personality sliders, runs them through
    /// the network in Sentis and draws its action from the scores. At lower
    /// difficulty levels it only decides every few steps and repeats its last
    /// action in between, exactly as the levels were rated in Python.
    ///
    /// It starts by itself when Play is pressed, in any scene (Bootstrap below).
    /// The menu ArenaMind > Play the levels against each other (StrengthCheck)
    /// checks that the NPC here is as strong as in Python.
    /// </summary>
    public class ArenaGame : MonoBehaviour
    {
        private const int You = 0, Npc = 1;
        private const float PanelWidth = 290f;  // GUI units
        private const float ResultPause = 2f;  // seconds the result stays up
        private const float CheckTolerance = 1e-4f;

        private static readonly string[] MoveNames =
            { "stand still", "forward", "forward-right", "right", "back-right", "back", "back-left", "left", "forward-left" };

        private GameConfig cfg;
        private Game game;
        private ArenaData data;
        private ArenaPolicy policy;
        private ArenaView view;
        private NpcBrain npc;
        private Camera cam;
        private readonly System.Random random = new System.Random();

        private int level, personality;
        private double clock;  // seconds since the last step
        private float pause;  // seconds left showing a result
        private string result = "";
        private int wins, losses, draws;
        private bool shootPressed, dashPressed;  // pressed at some point since the last step
        private bool showMemory = true;

        private string problem;  // why the game could not start
        private string checkText = "";
        private readonly double[] moveP = new double[9], shootP = new double[2], dashP = new double[2];
        private float guiScale = 1f;
        private GUIStyle panelStyle, titleStyle, textStyle, smallStyle, scoreStyle, resultStyle;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            if (FindAnyObjectByType<ArenaGame>() == null)
            {
                new GameObject("ArenaMind").AddComponent<ArenaGame>();
            }
        }

        private void Start()
        {
            cfg = new GameConfig();
            game = new Game(cfg);
            try
            {
                data = ArenaData.Load();
                if (data.observation != cfg.ObsDim || data.inputs != cfg.ObsDim + data.sliders.Length)
                {
                    throw new InvalidOperationException("arenamind.json is for observations of " + data.observation
                                                        + " numbers, but this game makes " + cfg.ObsDim);
                }
                policy = new ArenaPolicy(data.inputs);
                float worst = policy.Check(data);
                checkText = worst <= CheckTolerance
                    ? "same scores as PyTorch (largest difference " + worst.ToString("0.0e0") + " in "
                      + data.checks.Length + " checks)"
                    : "scores DIFFER from PyTorch by up to " + worst + ": export the policy again";
                Debug.Log("ArenaMind: Sentis gives " + checkText);
            }
            catch (Exception error)
            {
                problem = error.Message;
                Debug.LogError("ArenaMind could not start: " + error);
                return;
            }

            level = Mathf.Min(2, data.levels.Length - 1);  // start in the middle: L3
            personality = 0;
            for (int i = 0; i < data.personalities.Length; i++)
            {
                if (data.personalities[i].name == "neutral")
                {
                    personality = i;
                }
            }
            npc = new NpcBrain(policy, cfg, Npc, data.inputs, data.levels[level].every,
                               data.personalities[personality].sliders, random);
            view = new ArenaView(cfg);
            cam = Camera.main;
            if (cam == null)
            {
                cam = new GameObject("Main Camera").AddComponent<Camera>();
                cam.tag = "MainCamera";
            }
            cam.orthographic = true;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = ArenaView.Outside;
            NewGame();
        }

        private void OnDestroy()
        {
            if (policy != null)
            {
                policy.Dispose();
            }
        }

        private void Update()
        {
            if (problem != null)
            {
                return;
            }
            guiScale = Mathf.Max(1f, Screen.height / 900f);
            FitCamera();
            ReadPresses();
            if (pause > 0f)
            {
                pause -= Time.deltaTime;
                if (pause <= 0f)
                {
                    NewGame();
                }
            }
            else
            {
                clock += Time.deltaTime;
                int steps = 0;
                while (clock >= cfg.Dt && steps < 3 && !game.Done)
                {
                    clock -= cfg.Dt;
                    StepOnce();
                    steps++;
                }
                if (clock > cfg.Dt)
                {
                    clock = cfg.Dt;  // after a long frame, do not try to catch up further
                }
            }
            float blend = pause > 0f ? 1f : Mathf.Clamp01((float)(clock / cfg.Dt));
            view.Draw(game, blend, showMemory);
        }

        private void StepOnce()
        {
            PlayerAction mine = HumanAction();
            PlayerAction theirs = npc.Act(game);
            view.BeforeStep(game);
            game.Step(mine, theirs);
            if (game.Done)
            {
                if (game.Outcome > 0) { wins++; result = "You win"; }
                else if (game.Outcome < 0) { losses++; result = "The NPC wins"; }
                else { draws++; result = "Draw"; }
                pause = ResultPause;
                clock = 0.0;
            }
        }

        private void NewGame()
        {
            game.Reset(random);
            npc.NewGame();
            view.BeforeStep(game);
            clock = 0.0;
            pause = 0f;
            result = "";
        }

        private void Restart()
        {
            wins = 0;
            losses = 0;
            draws = 0;
            NewGame();
        }

        private void SetLevel(int k)
        {
            if (k >= 0 && k < data.levels.Length && k != level)
            {
                level = k;
                npc.Gate.Every = data.levels[level].every;
                Restart();
            }
        }

        private void SetPersonality(int k)
        {
            if (k >= 0 && k < data.personalities.Length && k != personality)
            {
                personality = k;
                npc.Sliders = data.personalities[personality].sliders;
                Restart();
            }
        }

        // The game view as a picture for the README, saved in the Python project's docs folder.
        private void SaveScreenshot()
        {
            string path = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "docs", "stage4_unity.png"));
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            ScreenCapture.CaptureScreenshot(path);
            Debug.Log("ArenaMind: saved a screenshot to " + path);
        }

        // ------------------------------------------------------------------
        // You
        // ------------------------------------------------------------------

        private Vector2 Pointer()
        {
#if ENABLE_INPUT_SYSTEM
            return Mouse.current != null ? Mouse.current.position.ReadValue() : Vector2.zero;
#else
            return Input.mousePosition;
#endif
        }

        private bool OverPanel(Vector2 pointer)
        {
            return pointer.x < (PanelWidth + 16f) * guiScale;
        }

        // Clicks and key presses between two steps still count at the next step.
        private void ReadPresses()
        {
            bool panel = OverPanel(Pointer());
#if ENABLE_INPUT_SYSTEM
            Keyboard keyboard = Keyboard.current;
            Mouse mouse = Mouse.current;
            if (mouse != null && !panel)
            {
                shootPressed |= mouse.leftButton.wasPressedThisFrame;
                dashPressed |= mouse.rightButton.wasPressedThisFrame;
            }
            if (keyboard == null)
            {
                return;
            }
            shootPressed |= keyboard.spaceKey.wasPressedThisFrame;
            dashPressed |= keyboard.leftShiftKey.wasPressedThisFrame || keyboard.rightShiftKey.wasPressedThisFrame;
            if (keyboard.rKey.wasPressedThisFrame) Restart();
            if (keyboard.tabKey.wasPressedThisFrame) showMemory = !showMemory;
            if (keyboard.pKey.wasPressedThisFrame) SaveScreenshot();
            if (keyboard.digit1Key.wasPressedThisFrame) SetLevel(0);
            if (keyboard.digit2Key.wasPressedThisFrame) SetLevel(1);
            if (keyboard.digit3Key.wasPressedThisFrame) SetLevel(2);
            if (keyboard.digit4Key.wasPressedThisFrame) SetLevel(3);
            if (keyboard.digit5Key.wasPressedThisFrame) SetLevel(4);
#else
            if (!panel)
            {
                shootPressed |= Input.GetMouseButtonDown(0);
                dashPressed |= Input.GetMouseButtonDown(1);
            }
            shootPressed |= Input.GetKeyDown(KeyCode.Space);
            dashPressed |= Input.GetKeyDown(KeyCode.LeftShift) || Input.GetKeyDown(KeyCode.RightShift);
            if (Input.GetKeyDown(KeyCode.R)) Restart();
            if (Input.GetKeyDown(KeyCode.Tab)) showMemory = !showMemory;
            if (Input.GetKeyDown(KeyCode.P)) SaveScreenshot();
            if (Input.GetKeyDown(KeyCode.Alpha1)) SetLevel(0);
            if (Input.GetKeyDown(KeyCode.Alpha2)) SetLevel(1);
            if (Input.GetKeyDown(KeyCode.Alpha3)) SetLevel(2);
            if (Input.GetKeyDown(KeyCode.Alpha4)) SetLevel(3);
            if (Input.GetKeyDown(KeyCode.Alpha5)) SetLevel(4);
#endif
        }

        private PlayerAction HumanAction()
        {
            Vector2 pointer = Pointer();
            bool panel = OverPanel(pointer);
            bool up, down, left, right, shoot, dash;
#if ENABLE_INPUT_SYSTEM
            Keyboard keyboard = Keyboard.current;
            Mouse mouse = Mouse.current;
            bool k = keyboard != null;
            up = k && (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed);
            down = k && (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed);
            left = k && (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed);
            right = k && (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed);
            shoot = (mouse != null && !panel && mouse.leftButton.isPressed) || (k && keyboard.spaceKey.isPressed);
            dash = (mouse != null && !panel && mouse.rightButton.isPressed)
                   || (k && (keyboard.leftShiftKey.isPressed || keyboard.rightShiftKey.isPressed));
#else
            up = Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow);
            down = Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow);
            left = Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow);
            right = Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow);
            shoot = (!panel && Input.GetMouseButton(0)) || Input.GetKey(KeyCode.Space);
            dash = (!panel && Input.GetMouseButton(1)) || Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
#endif
            shoot |= shootPressed;
            dash |= dashPressed;
            shootPressed = false;
            dashPressed = false;
            Vector2 aim = view.ScreenToGame(cam, pointer);
            double dx = (right ? 1 : 0) - (left ? 1 : 0);
            double dy = (down ? 1 : 0) - (up ? 1 : 0);  // the game's y grows down the screen
            return Controls.Human(game, You, aim.x, aim.y, dx, dy, shoot, dash);
        }

        // ------------------------------------------------------------------
        // Camera and panel
        // ------------------------------------------------------------------

        // The whole arena in view, to the right of the panel.
        private void FitCamera()
        {
            float size = (float)cfg.Size;
            float panel = (PanelWidth + 16f) * guiScale;  // pixels
            float height = Mathf.Max(1, Screen.height), width = Mathf.Max(panel + 1, Screen.width);
            cam.orthographicSize = Mathf.Max(size / 2 + 1f, (size + 2f) * height / (2f * (width - panel)));
            float panelWorld = panel * 2f * cam.orthographicSize / height;
            cam.transform.position = new Vector3(size / 2 - panelWorld / 2, size / 2, -10f);
        }

        private void MakeStyles()
        {
            if (titleStyle != null)
            {
                return;
            }
            var dark = new Texture2D(1, 1);
            dark.SetPixel(0, 0, new Color(0.16f, 0.16f, 0.15f, 0.94f));
            dark.Apply();
            panelStyle = new GUIStyle(GUI.skin.box) { padding = new RectOffset(10, 10, 10, 10) };
            panelStyle.normal.background = dark;
            titleStyle = new GUIStyle(GUI.skin.label) { fontSize = 20, fontStyle = FontStyle.Bold };
            textStyle = new GUIStyle(GUI.skin.label) { wordWrap = true };
            smallStyle = new GUIStyle(GUI.skin.label) { wordWrap = true, fontSize = 11 };
            scoreStyle = new GUIStyle(GUI.skin.label) { fontSize = 16, fontStyle = FontStyle.Bold };
            resultStyle = new GUIStyle(GUI.skin.box) { fontSize = 32, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };
        }

        // A row of buttons with the chosen one in the NPC's colour; returns the choice.
        private static int Choice(int selected, string[] names)
        {
            int chosen = selected;
            GUILayout.BeginHorizontal();
            for (int i = 0; i < names.Length; i++)
            {
                Color before = GUI.backgroundColor;
                if (i == selected)
                {
                    GUI.backgroundColor = ArenaView.Players[Npc];
                }
                if (GUILayout.Button(names[i]))
                {
                    chosen = i;
                }
                GUI.backgroundColor = before;
            }
            GUILayout.EndHorizontal();
            return chosen;
        }

        private void Bar(string label, double p)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(150));
            Rect r = GUILayoutUtility.GetRect(80, 18, GUILayout.Width(80));
            Color before = GUI.color;
            GUI.color = new Color(1f, 1f, 1f, 0.25f);
            GUI.DrawTexture(new Rect(r.x, r.y + 5, r.width, 8), Texture2D.whiteTexture);
            GUI.color = ArenaView.Players[Npc];
            GUI.DrawTexture(new Rect(r.x, r.y + 5, r.width * (float)p, 8), Texture2D.whiteTexture);
            GUI.color = before;
            GUILayout.Label((p * 100).ToString("0") + "%", GUILayout.Width(40));
            GUILayout.EndHorizontal();
        }

        private void OnGUI()
        {
            MakeStyles();
            GUI.matrix = Matrix4x4.Scale(new Vector3(guiScale, guiScale, 1f));
            if (problem != null)
            {
                GUI.Label(new Rect(20, 20, 700, 300), "ArenaMind could not start:\n\n" + problem, textStyle);
                return;
            }
            if (game == null || data == null)
            {
                return;
            }

            GUILayout.BeginArea(new Rect(8, 8, PanelWidth, Screen.height / guiScale - 16), panelStyle);
            GUILayout.Label("ArenaMind", titleStyle);
            GUILayout.Label("You (blue) against the trained NPC (orange).", textStyle);

            GUILayout.Space(8);
            GUILayout.Label("Difficulty (keys 1-5)");
            var levelNames = new string[data.levels.Length];
            for (int i = 0; i < levelNames.Length; i++)
            {
                levelNames[i] = data.levels[i].name;
            }
            SetLevel(Choice(level, levelNames));
            ArenaLevel lv = data.levels[level];
            float elo = personality < lv.elo.Length ? lv.elo[personality] : 0f;
            GUILayout.Label("Decides every " + lv.every.ToString("0.##") + " steps (" + (lv.every * cfg.Dt).ToString("0.00")
                            + " s). " + (elo > 0 ? "Rated " + elo.ToString("0") + " against the bots (rule bot = 1000)." : ""),
                            smallStyle);

            GUILayout.Label("Personality");
            var personalityNames = new string[data.personalities.Length];
            for (int i = 0; i < personalityNames.Length; i++)
            {
                personalityNames[i] = data.personalities[i].name;
            }
            SetPersonality(Choice(personality, personalityNames));
            string settings = data.personalities[personality].settings;
            GUILayout.Label(settings == "" ? "the policy's own play" : settings, smallStyle);

            GUILayout.Space(8);
            GUILayout.Label("You " + wins + " : " + losses + " NPC" + (draws > 0 ? "   (" + draws + " drawn)" : ""), scoreStyle);
            GUILayout.Label("HP: you " + game.Hp[You].ToString("0") + ", NPC " + game.Hp[Npc].ToString("0") + ".   "
                            + ((cfg.MaxSteps - game.T) * cfg.Dt).ToString("0") + " s left", textStyle);

            GUILayout.Space(8);
            GUILayout.Label("What the NPC is thinking", textStyle);
            if (npc.LastScores != null)
            {
                ActionSampler.Probabilities(npc.LastScores, 0, 9, moveP);
                ActionSampler.Probabilities(npc.LastScores, 9 + 5, 2, shootP);
                ActionSampler.Probabilities(npc.LastScores, 9 + 5 + 2, 2, dashP);
            }
            int top = 0;
            for (int i = 1; i < moveP.Length; i++)
            {
                if (moveP[i] > moveP[top]) top = i;
            }
            Bar("shoot", shootP[1]);
            Bar("dash", dashP[1]);
            Bar("move " + MoveNames[top], moveP[top]);
            GUILayout.Label(npc.Decisions + " decisions this game, " + npc.LastMilliseconds.ToString("0.00")
                            + " ms each in Sentis.", smallStyle);
            GUILayout.Label("Sentis: " + checkText + ".", smallStyle);

            GUILayout.FlexibleSpace();
            GUILayout.Label("WASD or arrows move, the mouse aims. Click or Space shoots, Shift or right click dashes. "
                            + "Tab shows where the NPC thinks you are (" + (showMemory ? "on" : "off") + "). "
                            + "R restarts, P saves a screenshot.",
                            smallStyle);
            GUILayout.EndArea();

            if (pause > 0f && result != "")
            {
                Vector3 centre = cam.WorldToScreenPoint(new Vector3((float)cfg.Size / 2, (float)cfg.Size / 2, 0f));
                float x = centre.x / guiScale, y = (Screen.height - centre.y) / guiScale;
                GUI.Box(new Rect(x - 160, y - 40, 320, 80), result, resultStyle);
            }
        }
    }
}
