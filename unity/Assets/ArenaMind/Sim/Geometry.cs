using System;

namespace ArenaMind.Sim
{
    /// <summary>
    /// The collision geometry of arena/geometry.py, for one point or ray at a
    /// time. Each function does the same arithmetic in the same order as the
    /// numpy version, so both give the same numbers. Rectangles are
    /// (x min, y min, x max, y max) one after another in a flat array.
    /// Functions that look along a path return the distance travelled until
    /// the first contact, or infinity if there is none.
    /// </summary>
    public static class Geometry
    {
        private const double Inf = double.PositiveInfinity;

        /// <summary>Length of a vector, as numpy.linalg.norm.</summary>
        public static double Length(double x, double y)
        {
            return Math.Sqrt(x * x + y * y);
        }

        /// <summary>
        /// Moves a circle (centre point[0], point[1]) so that it no longer
        /// overlaps any rectangle: away from the nearest point of a rectangle
        /// it touches, or out through the nearest edge if its centre is inside.
        /// </summary>
        public static void PushOutOfRects(double[] point, double radius, double[] rects)
        {
            for (int r = 0; r + 3 < rects.Length; r += 4)
            {
                double x0 = rects[r], y0 = rects[r + 1], x1 = rects[r + 2], y1 = rects[r + 3];
                double x = point[0], y = point[1];
                double nx = Math.Min(Math.Max(x, x0), x1);  // nearest point of the rectangle
                double ny = Math.Min(Math.Max(y, y0), y1);
                double dx = x - nx, dy = y - ny;
                double dist = Length(dx, dy);
                double newX = x, newY = y;
                if (dist > 0)
                {
                    if (dist < radius)
                    {
                        newX = nx + dx / dist * radius;
                        newY = ny + dy / dist * radius;
                    }
                }
                else
                {
                    // Inside: leave through the nearest edge (the first one on a tie).
                    int side = 0;
                    double best = x - x0;
                    if (x1 - x < best) { side = 1; best = x1 - x; }
                    if (y - y0 < best) { side = 2; best = y - y0; }
                    if (y1 - y < best) { side = 3; }
                    if (side == 0) { newX = x0 - radius; }
                    else if (side == 1) { newX = x1 + radius; }
                    else if (side == 2) { newY = y0 - radius; }
                    else { newY = y1 + radius; }
                }
                point[0] = newX;
                point[1] = newY;
            }
        }

        /// <summary>Keeps a coordinate of a circle's centre at least radius from the arena walls.</summary>
        public static double ClampToArena(double v, double radius, double size)
        {
            return Math.Min(Math.Max(v, radius), size - radius);
        }

        /// <summary>
        /// Distance along a ray (unit direction dx, dy) to a rectangle: 0 if it
        /// starts inside, infinity if it misses. Slab method: the ray is within
        /// the rectangle's x range for one interval of distances and within its
        /// y range for another; it is inside the rectangle where both overlap.
        /// </summary>
        public static double RayRect(double ox, double oy, double dx, double dy,
                                     double x0, double y0, double x1, double y1)
        {
            double nearX, farX, nearY, farY;
            if (dx == 0)
            {
                bool inside = ox >= x0 && ox <= x1;  // parallel: inside for every distance, or never
                nearX = inside ? -Inf : Inf;
                farX = inside ? Inf : -Inf;
            }
            else
            {
                double inv = 1.0 / dx;
                double t1 = (x0 - ox) * inv;
                double t2 = (x1 - ox) * inv;
                nearX = Math.Min(t1, t2);
                farX = Math.Max(t1, t2);
            }
            if (dy == 0)
            {
                bool inside = oy >= y0 && oy <= y1;
                nearY = inside ? -Inf : Inf;
                farY = inside ? Inf : -Inf;
            }
            else
            {
                double inv = 1.0 / dy;
                double t1 = (y0 - oy) * inv;
                double t2 = (y1 - oy) * inv;
                nearY = Math.Min(t1, t2);
                farY = Math.Max(t1, t2);
            }
            double enter = Math.Max(nearX, nearY);
            double exit = Math.Min(farX, farY);
            return (enter <= exit && exit >= 0) ? Math.Max(enter, 0.0) : Inf;
        }

        /// <summary>Distance along a ray to the nearest rectangle.</summary>
        public static double RayRects(double ox, double oy, double dx, double dy, double[] rects)
        {
            double best = Inf;
            for (int r = 0; r + 3 < rects.Length; r += 4)
            {
                best = Math.Min(best, RayRect(ox, oy, dx, dy, rects[r], rects[r + 1], rects[r + 2], rects[r + 3]));
            }
            return best;
        }

        /// <summary>Distance along a ray that starts inside the arena to the wall it hits.</summary>
        public static double RayArenaExit(double ox, double oy, double dx, double dy, double size)
        {
            double tx = dx > 0 ? (size - ox) / dx : (dx < 0 ? -ox / dx : Inf);
            double ty = dy > 0 ? (size - oy) / dy : (dy < 0 ? -oy / dy : Inf);
            return Math.Min(tx, ty);
        }

        /// <summary>Range sensor: distance to the nearest cover or wall, at most maxDist.</summary>
        public static double Raycast(double ox, double oy, double dx, double dy, double[] rects,
                                     double size, double maxDist)
        {
            return Math.Min(Math.Min(RayRects(ox, oy, dx, dy, rects), RayArenaExit(ox, oy, dx, dy, size)), maxDist);
        }

        /// <summary>Whether nothing in rects blocks the straight line from a to b.</summary>
        public static bool LineOfSight(double ax, double ay, double bx, double by, double[] rects)
        {
            double vx = bx - ax, vy = by - ay;
            double length = Length(vx, vy);
            double safe = length > 0 ? length : 1.0;
            return RayRects(ax, ay, vx / safe, vy / safe, rects) >= length;
        }

        /// <summary>Distance from p0 to the first rectangle on the segment p0 to p1.</summary>
        public static double SegmentRectsEntry(double x0, double y0, double x1, double y1, double[] rects)
        {
            double vx = x1 - x0, vy = y1 - y0;
            double length = Length(vx, vy);
            double safe = length > 0 ? length : 1.0;
            double dist = RayRects(x0, y0, vx / safe, vy / safe, rects);
            return dist <= length ? dist : Inf;
        }

        /// <summary>
        /// Distance from p0 to where the segment p0 to p1 first touches a
        /// circle: the smaller root t of |p0 + t (p1 - p0) - centre| = radius,
        /// if it lies in [0, 1], times the segment's length.
        /// </summary>
        public static double SegmentCircleEntry(double x0, double y0, double x1, double y1,
                                                double cx, double cy, double radius)
        {
            double dx = x1 - x0, dy = y1 - y0;
            double fx = x0 - cx, fy = y0 - cy;
            double a = dx * dx + dy * dy;
            double b = 2.0 * (fx * dx + fy * dy);
            double c = (fx * fx + fy * fy) - radius * radius;
            double disc = b * b - 4.0 * a * c;
            double t = (-b - Math.Sqrt(Math.Max(disc, 0.0))) / (2.0 * a);
            bool startsInside = c <= 0;
            bool crosses = a > 0 && disc >= 0 && t >= 0 && t <= 1;
            if (startsInside)
            {
                t = 0.0;
            }
            return (startsInside || crosses) ? t * Math.Sqrt(a) : Inf;
        }

        /// <summary>Whether a point lies in any rectangle (edges included).</summary>
        public static bool InRects(double x, double y, double[] rects)
        {
            for (int r = 0; r + 3 < rects.Length; r += 4)
            {
                if (x >= rects[r] && x <= rects[r + 2] && y >= rects[r + 1] && y <= rects[r + 3])
                {
                    return true;
                }
            }
            return false;
        }
    }
}
