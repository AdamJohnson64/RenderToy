////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using System.Threading;

namespace RenderToy
{
    [TestClass]
    public class ClipHelpTests
    {
        [TestMethod]
        public void ClipTriangle3DTest()
        {
            var triangle = new Triangle3D(
                new Vector3D(0, 0, 0),
                new Vector3D(0, 1, 0),
                new Vector3D(1, 0, 0));
            // Clip right side offset 0.5 for a single triangle.
            {
                var clipped = ClipHelp.ClipTriangle3D(triangle, new Vector3D(1, 0, 0), 0.5);
                // There should be one triangle.
                if (clipped.Count() != 1) throw new Exception("N=[1,0,0], d=0.5; Expected one triangle.");
                // There should be no X coordinate below +0.5.
                var vertices = clipped.SelectMany(t => new[] { t.P0, t.P1, t.P2 });
                if (vertices.Any(v => v.X < 0.5)) throw new Exception("N=[1,0,0], d=0.5; vertex behind clip plane?");
            }
            // Clip left side offset 0.5 for two triangles.
            {
                var clipped = ClipHelp.ClipTriangle3D(triangle, new Vector3D(-1, 0, 0), -0.5);
                if (clipped.Count() != 2) throw new Exception("N=[-1,0,0], d=-0.5; Expected two triangles.");
                // There should be no X coordinate above +0.5.
                var vertices = clipped.SelectMany(t => new[] { t.P0, t.P1, t.P2 });
                if (vertices.Any(v => v.X > 0.5)) throw new Exception("N=[-1,0,0], d=-0.5; vertex behind clip plane?");
            }
            // Trivial pass; no clipping.
            {
                var clipped = ClipHelp.ClipTriangle3D(triangle, new Vector3D(1, 0, 0), -2);
                if (clipped.Count() != 1) throw new Exception("Expected one triangle.");
                if (!Triangle3DEqual(clipped.First(), triangle)) throw new Exception("N=[1,0,0], d=-2; Unclipped triangle was modified?");
            }
            // Trivial clip; complete clipping.
            {
                var clipped = ClipHelp.ClipTriangle3D(triangle, new Vector3D(1, 0, 0), 2);
                if (clipped.Count() != 0) throw new Exception("Expected no triangles.");
            }
        }

        static bool Triangle3DEqual(Triangle3D lhs, Triangle3D rhs)
        {
            return Vector3DEqual(lhs.P0, rhs.P0) && Vector3DEqual(lhs.P1, rhs.P1) && Vector3DEqual(lhs.P2, rhs.P2);
        }
        static bool Vector3DEqual(Vector3D lhs, Vector3D rhs)
        {
            return lhs.X == rhs.X && lhs.Y == rhs.Y && lhs.Z == rhs.Z;
        }
    }
    [TestClass]
    public class MeshPLYTests
    {
        [TestMethod]
        public void LoadAllPLYFilesTest()
        {
            string mydocuments = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            string[] all_ply_files = Directory.GetFiles(mydocuments, "*.ply");
            var results = all_ply_files
                .Select(filename => Path.Combine(mydocuments, filename))
                .Select(pathname => {
                    try
                    {
                        Console.WriteLine("Loading mesh '" + pathname + "'.");
                        return FileFormat.LoadPLYFromPath(pathname);
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Exception while loading '" + pathname + "': " + e.Message);
                        return null;
                    }
                }).ToArray();
            if (results.Any(x => x == null))
            {
                throw new Exception("There were errors processing some files; refer to output for details.");
            }
        }
    }
    [TestClass]
    public class WorkQueueTests
    {
        [TestMethod]
        public void WorkQueueSurge()
        {
            var work = new WorkQueue();
            // Add lots of dummy work.
            Console.WriteLine("Creating initial task load.");
            for (int i = 0; i < 1000000; ++i)
            {
                int j = i;
                work.Queue(() =>
                {
                    for (int dummy = 0; dummy < 10000; ++dummy) ;
                    // The last work item will wait a while and then throw a large amount of work into the queue.
                    // This ensures that workers are not prematurely exiting reducing late throughput.
                    if (j == 999999)
                    {
                        Console.WriteLine("Initial task waiting.");
                        Wait(TimeSpan.FromMilliseconds(5000));
                        Console.WriteLine("Creating late task surge.");
                        for (int surge = 0; surge < 10000; ++surge)
                        {
                            work.Queue(() =>
                            {
                                for (int dummy = 0; dummy < 1000000; ++dummy) ;
                            });
                        }
                    }
                }
                );
            }
            // Start running work.
            work.Start();
        }
        public static void Wait(TimeSpan waitfor)
        {
            Console.WriteLine("Start waiting for " + waitfor + ".");
            DateTime start = DateTime.Now;
        AGAIN:
            DateTime end = DateTime.Now;
            if (end.Subtract(start).CompareTo(waitfor) < 0)
            {
                Thread.Sleep(1);
                goto AGAIN;
            }
            Console.WriteLine("End waiting for " + waitfor + ".");
        }
    }
}
