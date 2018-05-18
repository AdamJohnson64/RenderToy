////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using Microsoft.VisualStudio.TestTools.UnitTesting;
using RenderToy.BoundingVolumeHierarchy;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.PipelineModel;
using RenderToy.Primitives;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

namespace RenderToy
{
    [TestClass]
    public class BVHTests
    {
        [TestMethod]
        public void BVHTimingTest()
        {
            var mesh = Mesh.CreateMesh(new Sphere(), 100, 100);
            var triangles = Triangle3D.ExtractTriangles(mesh.Vertices, mesh.Triangles).ToArray();
            Performance.LogBegin("BVH Octree (Reference)");
            try
            {
                var node = BoundingVolumeHierarchy.OctreeREF.Create(triangles);
                VerifyMesh(triangles, node);
            }
            catch (Exception e)
            {
                Performance.LogEvent("Exception while calculating BVH: " + e.Message);
            }
            Performance.LogEnd("BVH Octree (Reference)");
            Performance.LogBegin("BVH Octree (Baseline)");
            try
            {
                var node = BoundingVolumeHierarchy.Octree.Create(triangles);
                VerifyMesh(triangles, node);
            }
            catch (Exception e)
            {
                Performance.LogEvent("Exception while calculating BVH: " + e.Message);
            }
            Performance.LogEnd("BVH Octree (Baseline)");
            Performance.LogBegin("BVH KD (Baseline)");
            try
            {
                var node = BoundingVolumeHierarchy.KDTree.Create(triangles);
                VerifyMesh(triangles, node);
            }
            catch (Exception e)
            {
                Performance.LogEvent("Exception while calculating BVH: " + e.Message);
            }
            Performance.LogEnd("BVH KD (Baseline)");
        }
        static void VerifyMesh(Triangle3D[] triangles, MeshBVH root)
        {
            // Run some sanity checks on the mesh.
            // All subnode bounds should be completely contained by their parent.
            var allnode = EnumerateNodes(root).ToArray();
            if (!allnode
                .Where(n => n.Parent != null)
                .All(n => CommonBVH.ShapeContains(n.Parent.Bound, n.Node.Bound)))
            {
                throw new InvalidDataException("A child node exists which is not contained by its parent.");
            }
            // All node bounds should at least contain all their triangles.
            if (!allnode
                .Where(n => n.Node.Triangles != null)
                .All(n => CommonBVH.ShapeContains(n.Node.Bound, CommonBVH.ComputeBounds(n.Node.Triangles))))
            {
                throw new InvalidDataException("A node bound exists which does not contain its triangle extents.");
            }
            // Make sure we don't exceed the maximum depth for the BVH.
            int maximum_bvh_depth = allnode.Max(x => x.Level);
            if (maximum_bvh_depth > CommonBVH.MAXIMUM_BVH_DEPTH)
            {
                throw new InvalidDataException("This BVH is " + maximum_bvh_depth + " levels deep; this exceeds the maximum depth and will fail on GPU.");
            }
        }
        static IEnumerable<NodeParent> EnumerateNodes(MeshBVH root)
        {
            return EnumerateNodes(root, null, 0);
        }
        static IEnumerable<NodeParent> EnumerateNodes(MeshBVH node, MeshBVH parent, int level)
        {
            yield return new NodeParent(node, parent, level);
            if (node.Children != null)
            {
                foreach (var child in node.Children)
                {
                    foreach (var subnode in EnumerateNodes(child, node, level + 1))
                    {
                        yield return subnode;
                    }
                }
            }
        }
        struct NodeParent
        {
            public NodeParent(MeshBVH node, MeshBVH parent, int level)
            {
                Node = node;
                Parent = parent;
                Level = level;
            }
            public readonly MeshBVH Node;
            public readonly MeshBVH Parent;
            public readonly int Level;
        }
    }
    [TestClass]
    public class ClipHelpTests
    {
        [TestMethod]
        public void ClipTriangle3DTest()
        {
            var triangle = new Vector3D[]
            {
                new Vector3D(0, 0, 0),
                new Vector3D(0, 1, 0),
                new Vector3D(1, 0, 0),
            };
            // Clip right side offset 0.5 for a single triangle.
            {
                var clipped = Clipping.ClipTriangle(triangle, new Vector3D(1, 0, 0), 0.5);
                // There should be one triangle.
                if (clipped.Count() != 3 * 1) throw new Exception("N=[1,0,0], d=0.5; Expected one triangle.");
                // There should be no X coordinate below +0.5.
                if (clipped.Any(v => v.X < 0.5)) throw new Exception("N=[1,0,0], d=0.5; vertex behind clip plane?");
            }
            // Clip left side offset 0.5 for two triangles.
            {
                var clipped = Clipping.ClipTriangle(triangle, new Vector3D(-1, 0, 0), -0.5);
                if (clipped.Count() != 3 * 2) throw new Exception("N=[-1,0,0], d=-0.5; Expected two triangles.");
                // There should be no X coordinate above +0.5.
                if (clipped.Any(v => v.X > 0.5)) throw new Exception("N=[-1,0,0], d=-0.5; vertex behind clip plane?");
            }
            // Trivial pass; no clipping.
            {
                var clipped = Clipping.ClipTriangle(triangle, new Vector3D(1, 0, 0), -2);
                if (clipped.Count() != 3 * 1) throw new Exception("Expected one triangle.");
                if (!clipped.SequenceEqual(triangle)) throw new Exception("N=[1,0,0], d=-2; Unclipped triangle was modified?");
            }
            // Trivial clip; complete clipping.
            {
                var clipped = Clipping.ClipTriangle(triangle, new Vector3D(1, 0, 0), 2);
                if (clipped.Count() != 3 * 0) throw new Exception("Expected no triangles.");
            }
        }
        static bool Vector3DEqual(Vector3D lhs, Vector3D rhs)
        {
            return lhs.X == rhs.X && lhs.Y == rhs.Y && lhs.Z == rhs.Z;
        }
    }
    [TestClass]
    public class MeshPLYTests
    {
        static void ForAllTestModels(string wildcard, Action<string> execute)
        {
            var root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "RenderToyModels");
            var results = Directory.EnumerateFiles(root, wildcard, SearchOption.AllDirectories)
                .Select(filename => Path.Combine(root, filename))
                .Select(pathname => {
                    try
                    {
                        Performance.LogEvent("Loading mesh '" + pathname + "'.");
                        execute(pathname);
                        return true;
                    }
                    catch (Exception e)
                    {
                        Performance.LogEvent("Exception while loading '" + pathname + "': " + e.Message);
                        return false;
                    }
                }).ToArray();
            if (results.Any(x => !x))
            {
                throw new Exception("There were errors processing some files; refer to output for details.");
            }
        }
        [TestMethod]
        public void LoadAllBPTFilesTest()
        {
            ForAllTestModels("*.bpt", (pathname) => LoaderBPT.LoadFromPath(pathname));
        }
        [TestMethod]
        public void LoadAllOBJFilesTest()
        {
            ForAllTestModels("*.obj", (pathname) => LoaderOBJ.LoadFromPath(pathname));
        }
        [TestMethod]
        public void LoadAllPLYFilesTest()
        {
            ForAllTestModels("*.ply", (pathname) => LoaderPLY.LoadFromPath(pathname));
        }
    }
    [TestClass]
    public class WorkQueueTests
    {
        [TestMethod]
        public void WorkQueueSurge()
        {
            Performance.LogBegin("WorkQueueSurge Test");
            var work = new WorkQueue();
            // Add lots of dummy work.
            Performance.LogEvent("Creating initial task load");
            for (int i = 0; i < 100; ++i)
            {
                int j = i;
                work.Queue(() =>
                {
                    Performance.LogBegin("Initial Task " + j);
                    Wait(TimeSpan.FromMilliseconds(100));
                    // The last work item will wait a while and then throw a large amount of work into the queue.
                    // This ensures that workers are not prematurely exiting reducing late throughput.
                    if (j == 99)
                    {
                        Performance.LogEvent("Initial task waiting");
                        Wait(TimeSpan.FromMilliseconds(1000));
                        Performance.LogEvent("Creating late surge");
                        for (int surge = 0; surge < 100; ++surge)
                        {
                            int j2 = surge;
                            work.Queue(() =>
                            {
                                Performance.LogBegin("Surge Task " + j2);
                                Wait(TimeSpan.FromMilliseconds(100));
                                Performance.LogEnd("Surge Task " + j2);
                            });
                        }
                    }
                    Performance.LogEnd("Initial Task " + j);
                }
                );
            }
            // Start running work.
            work.Start();
            Performance.LogEnd("WorkQueueSurge Test");
        }
        public static void Wait(TimeSpan waitfor)
        {
            DateTime start = DateTime.Now;
        AGAIN:
            DateTime end = DateTime.Now;
            if (end.Subtract(start).CompareTo(waitfor) < 0)
            {
                Thread.Sleep(1);
                goto AGAIN;
            }
        }
    }
}
