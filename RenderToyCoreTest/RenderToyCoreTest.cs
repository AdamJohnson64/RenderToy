////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using Microsoft.VisualStudio.TestTools.UnitTesting;
using RenderToy.BoundingVolumeHierarchy;
using RenderToy.Cameras;
using RenderToy.Diagnostics;
using RenderToy.Expressions;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.PipelineModel;
using RenderToy.Primitives;
using RenderToy.QueryEngine;
using RenderToy.SceneGraph;
using RenderToy.TextureFormats;
using RenderToy.Transforms;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Threading;

namespace RenderToy
{
    public static class TestUtil
    {
        public static void AllAssets(string wildcard, Action<string> execute)
        {
            var rootassembly = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            var rootassets = Path.Combine(rootassembly, "..\\..\\ThirdParty\\RenderToyAssets");
            var results = Directory.EnumerateFiles(rootassets, wildcard, SearchOption.AllDirectories)
                .Select(filename => Path.Combine(rootassets, filename))
                .Select(pathname =>
                {
                    try
                    {
                        Performance.LogEvent("Loading '" + pathname + "'.");
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
    }
    [TestClass]
    public class BVHTests
    {
        [TestMethod]
        public void BVHTimingTest()
        {
            var mesh = Mesh.CreateMesh(Sphere.Default, 100, 100);
            var triangles = Triangle3D.ExtractTriangles(mesh.Vertices.GetVertices(), mesh.Vertices.GetIndices()).ToArray();
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
    public class BVHChainTests
    {
        [TestMethod]
        public void BVHChainCompleteness()
        {
            var root = new MeshBVH(new Bound3D(new Vector3D(0,0,0), new Vector3D(4,4,0)), null, new[] {
                new MeshBVH(new Bound3D(new Vector3D(0,0,0), new Vector3D(2,2,0)), null, new[] {
                    new MeshBVH(new Bound3D(new Vector3D(0,0,0), new Vector3D(1,1,0)), null, null),
                    new MeshBVH(new Bound3D(new Vector3D(1,0,0), new Vector3D(2,1,0)), null, null),
                    new MeshBVH(new Bound3D(new Vector3D(0,1,0), new Vector3D(1,2,0)), null, null),
                    new MeshBVH(new Bound3D(new Vector3D(1,1,0), new Vector3D(2,2,0)), null, null),
                }),
                new MeshBVH(new Bound3D(new Vector3D(2,0,0), new Vector3D(2,2,0)), null, null),
                new MeshBVH(new Bound3D(new Vector3D(0,2,0), new Vector3D(2,2,0)), null, null),
                new MeshBVH(new Bound3D(new Vector3D(2,2,0), new Vector3D(2,2,0)), null, null)
            });
            var chain = MeshBVHChain.Create(root);
            var discovered = new HashSet<MeshBVHChain>();
            var walk = chain;
            while (walk != null)
            {
                Console.WriteLine("[" + walk.Bound.Min.X + "," + walk.Bound.Min.Y + "],[" + walk.Bound.Max.X + "," + walk.Bound.Max.Y + "]");
                if (discovered.Contains(walk))
                {
                    throw new Exception("Loop in BVH node chain.");
                }
                discovered.Add(walk);
                if (walk.Child != null)
                {
                    walk = walk.Child;
                }
                else
                {
                    walk = walk.Sibling;
                }
            }
            if (discovered.Count != 9)
            {
                throw new Exception("Expected to find 9 total nodes in the BVH node chain.");
            }
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
    public class CompilerTests
    {
        [TestMethod]
        public void PipelineCompileRenderer()
        {
            var renderer = Compiler.GenerateRenderer();
            Console.WriteLine(Compiler.DoString(renderer));
            Compiler.DoCompile(renderer);
        }
    }
    [TestClass]
    public class ExpressionTests
    {
        [TestMethod]
        public void CompileGenericHLSLVS50()
        {
            StockMaterials.PlasticWhite.CompileHLSL("vs", "vs_5_0");
        }
        [TestMethod]
        public void CompileGenericHLSLPS50()
        {
            StockMaterials.PlasticWhite.CompileHLSL("ps", "ps_5_0");
        }
        [TestMethod]
        public void GenerateBrickHLSL()
        {
            Console.WriteLine(StockMaterials.BrickAlbedo.GenerateHLSL());
        }
        [TestMethod]
        public void CompileBrickHLSLVS50()
        {
            StockMaterials.BrickAlbedo.CompileHLSL("vs", "vs_5_0");
        }
        [TestMethod]
        public void CompileBrickHLSLPS50()
        {
            StockMaterials.BrickAlbedo.CompileHLSL("ps", "ps_5_0");
        }
        [TestMethod]
        public void CompileBrickMSIL()
        {
            StockMaterials.BrickAlbedo.CompileMSIL();
        }
        [TestMethod]
        public void GenerateMarbleTileHLSL()
        {
            Console.WriteLine(StockMaterials.MarbleTile.GenerateHLSL());
        }
        [TestMethod]
        public void CompileMarbleTileHLSLVS50()
        {
            StockMaterials.MarbleTile.CompileHLSL("vs", "vs_5_0");
        }
        [TestMethod]
        public void CompileMarbleTileHLSLPS50()
        {
            StockMaterials.MarbleTile.CompileHLSL("ps", "ps_5_0");
        }
        [TestMethod]
        public void CompileMarbleTileMSIL()
        {
            StockMaterials.MarbleTile.CompileMSIL();
        }
    }
    [TestClass]
    public class ExpressionFlattenTests
    {
        static Expression<Func<double, double>> Number = (a) => a;
        static ExpressionFlatten<Func<double, double>> NumberFn = Number.Flatten();
        static Expression<Func<double, double, double>> Compound = (a, b) => NumberFn.Call(a) + NumberFn.Call(b);
        static ExpressionFlatten<Func<double, double, double>> CompoundFn = Compound.Flatten();
        [TestMethod]
        public void ExpressionFlattenTest()
        {
            Debug.Assert(CompoundFn.Call(1, 2) == 3);
        }
    }
    [TestClass]
    public class GeomTests
    {
        [TestMethod]
        public void GeomSphereTBNTest()
        {
            var sphere = Sphere.Default;
            for (int iv = 1; iv <= 9; ++iv)
            {
                for (int iu = 0; iu <= 10; ++iu)
                {
                    var tu = iu / 10.0;
                    var tv = iv / 10.0;
                    var pu0 = sphere.GetPointUV(tu - 0.01, tv);
                    var pu1 = sphere.GetPointUV(tu + 0.01, tv);
                    var TAN_APPROX = MathHelp.Normalized(pu1 - pu0);
                    var TAN_EXACT = sphere.GetTangentUV(tu, tv);
                    Debug.Assert(MathHelp.Dot(TAN_APPROX, TAN_EXACT) > 0.9);
                    var pv0 = sphere.GetPointUV(tu, tv - 0.01);
                    var pv1 = sphere.GetPointUV(tu, tv + 0.01);
                    var BIN_APPROX = MathHelp.Normalized(pv1 - pv0);
                    var BIN_EXACT = sphere.GetBitangentUV(tu, tv);
                    Debug.Assert(MathHelp.Dot(BIN_APPROX, BIN_EXACT) > 0.9);
                }
            }
        }
    }
    [TestClass]
    public class MathTests
    {
        [TestMethod]
        public void MathMatrixDeterminantIdentity()
        {
            var test = MatrixExtensions.Identity(4).Determinant();
            Console.WriteLine(test);
            var constant = (ConstantExpression)test;
            Debug.Assert(((double)constant.Value) == 1);
        }
        [TestMethod]
        public void MathMatrixDeterminantDX43()
        {
            var test = MatrixExtensions.CreateDX43().Determinant();
            Console.WriteLine("Debug:  " + test);
            Console.WriteLine("Pretty: " + test.PrettyPrint());
        }
        [TestMethod]
        public void MathMatrixDeterminantDX44()
        {
            var test = MatrixExtensions.CreateDX44().Determinant();
            Console.WriteLine("Debug:  " + test);
            Console.WriteLine("Pretty: " + test.PrettyPrint());
        }
        [TestMethod]
        public void MathMatrixInvertDX43()
        {
            MatrixExtensions.CreateDX43().Invert();
        }
        [TestMethod]
        public void MathMatrixInvertDX44()
        {
            MatrixExtensions.CreateDX44().Invert();
        }
        [TestMethod]
        public void MathMatrixInvertIdentity()
        {
            MatrixExtensions.Identity(4).Invert();
        }
        [TestMethod]
        public void MathMatrixTransposeIdentity()
        {
            MatrixExtensions.Identity(4).Transpose();
        }
    }
    [TestClass]
    public class MeshTests
    {
        [TestMethod]
        public void LoadAllBPTFilesTest()
        {
            TestUtil.AllAssets("*.bpt", async (pathname) => await LoaderBPT.LoadFromPathAsync(pathname));
        }
        [TestMethod]
        public void LoadAllOBJFilesTest()
        {
            TestUtil.AllAssets("*.obj", async (pathname) => await LoaderOBJ.LoadFromPathAsync(pathname));
        }
        [TestMethod]
        public void LoadAllPLYFilesTest()
        {
            TestUtil.AllAssets("*.ply", async (pathname) => await LoaderPLY.LoadFromPathAsync(pathname));
        }
    }
    [TestClass]
    public class PipelineQueryTests
    {
        [TestMethod]
        public void PipelineQueryEquivalence()
        {
            var pipe1 = triangles.Select(v => oldmethod(v));
            var pipe2 = triangles.Select(v => newmethod(v));
            Debug.Assert(pipe1.SequenceEqual(pipe2));
        }
        [TestMethod]
        public void PipelineQueryPerformance()
        {
            // Call directly into the transform pipe function.
            var time1 = Performance.Time(() =>
            {
                foreach (var v in triangles.Select(v => oldmethod)) { }
            });
            Console.WriteLine(time1);
            // Call into the rebuilt transform pipe function.
            var time2 = Performance.Time(() =>
            {
                foreach (var v in triangles.Select(v => newmethod)) { }
            });
            Console.WriteLine(time2);
            Debug.Assert(time1 > time2);
        }
        static IEnumerable<Vector3D> triangles = PrimitiveAssembly.CreateTriangles((IParametricUV)Sphere.Default, 2000, 2000);
        static Matrix3D mvp = Perspective.CreateProjection(0.01, 100.0, 60.0 * System.Math.PI / 180.0, 60.0 * System.Math.PI / 180.0);
        static Expression<Func<Vector3D, Vector4D>> TestExpressionFn = (v) =>
            Transformation.TransformToScreen.Call(
                Transformation.HomogeneousDivide.Call(
                    Transformation.Vector3ToVector4.Call(
                        MathHelp.TransformPoint(mvp, v))), 256, 256);
        static ExpressionFlatten<Func<Vector3D, Vector4D>> TestExpression = TestExpressionFn.ReplaceCalls().Flatten();
        static Func<Vector3D, Vector4D> oldmethod = TestExpressionFn.Compile();
        static Func<Vector3D, Vector4D> newmethod = TestExpression.Call;
    }
    [TestClass]
    public class QueryTests
    {
        [TestMethod]
        public void QueryEquivalenceImmediate()
        {
            var data = "Data";
            var query1 = Query.Create(data);
            var query2 = Query.Create(data);
            Debug.Assert(query1.Equals(query2));
        }
        [TestMethod]
        public void QueryNonEquivalenceImmediate()
        {
            var query1 = Query.Create("Data1");
            var query2 = Query.Create("Data2");
            Debug.Assert(!query1.Equals(query2));
        }
        [TestMethod]
        public void QueryEquivalenceDeferred()
        {
            var querydata = Query.Create("Data");
            var query1 = Query.Create((x) => x, querydata);
            var query2 = Query.Create((x) => x, querydata);
            Debug.Assert(query1.Equals(query2));
        }
        [TestMethod]
        public void QueryNonEquivalenceDeferred()
        {
            var querydata1 = Query.Create("Data1");
            var querydata2 = Query.Create("Data2");
            Debug.Assert(!querydata1.Equals(querydata2));
            var query1 = Query.Create((x) => x, querydata1);
            var query2 = Query.Create((x) => x, querydata2);
            Debug.Assert(!query1.Equals(query2));
        }
        [TestMethod]
        public void QueryNotifyDeferred()
        {
            var waitquery = new ManualResetEvent(false);
            var query = Query.Create(() => { waitquery.WaitOne(); return "Done"; });
            int triggercount = 0;
            query.AddListener(() => { Interlocked.Increment(ref triggercount); });
            Debug.Assert(triggercount == 1);
            waitquery.Set();
            while (triggercount < 2)
            {
                Thread.Sleep(1);
            }
            Debug.Assert(triggercount == 2);
        }
        [TestMethod]
        public void QueryNotifyImmediate()
        {
            var query = Query.Create("Done");
            int triggercount = 0;
            query.AddListener(() => { Interlocked.Increment(ref triggercount); });
            Debug.Assert(triggercount == 1);
        }
        [TestMethod]
        public void QueryResultDeferred()
        {
            var result = "Done";
            var waitquery = new AutoResetEvent(false);
            var query = Query.Create(() => { waitquery.WaitOne(); return result; });
            Debug.Assert(query.Result == null);
            waitquery.Set();
            while (query.Result != result)
            {
                Thread.Sleep(1);
            }
        }
        [TestMethod]
        public void QueryResultImmediate()
        {
            var result = "Done";
            var query = Query.Create(result);
            Debug.Assert(query.Result == result);
        }
        [TestMethod]
        public void QueryLongRunning()
        {
            const int SPHERECOUNT = 100000;
            Debug.WriteLine("Generating fake scene");
            var scene = new Scene();
            var transform = new TransformMatrix(Matrix3D.Identity);
            for (int i = 0; i < SPHERECOUNT; ++i)
            scene.children.Add(new Node("Sphere " + i, transform, Sphere.Default, StockMaterials.White, StockMaterials.PlasticWhite));
            Debug.WriteLine("Creating immediate scene query.");
            var queryscene = Query.Create(scene);
            var waitresult = new AutoResetEvent(false);
            Debug.WriteLine("Creating deferred transformed scene query.");
            var querytransformed = Query.Create((input) => (IReadOnlyList<TransformedObject>)TransformedObject.Enumerate(input).ToArray(), queryscene);
            querytransformed.AddListener(() => waitresult.Set());
            waitresult.Reset();
            Debug.Assert(querytransformed.Result == null);
            Debug.WriteLine("Waiting for query ready.");
            bool waitcomplete = waitresult.WaitOne(10000);
            Debug.Assert(waitcomplete);
            Debug.Assert(querytransformed.Result != null);
            Debug.Assert(querytransformed.Result.Count == SPHERECOUNT);
        }
    }
    [TestClass]
    public class TextureTests
    {
        [TestMethod]
        public void TextureLoadAllHDR()
        {
            TestUtil.AllAssets("*.hdr", (filename) => LoaderHDR.LoadFromPath(filename));
        }
        [TestMethod]
        public void TextureLoadAllPNG()
        {
            TestUtil.AllAssets("*.png", (filename) => LoaderPNG.LoadFromPath(filename));
        }
        [TestMethod]
        public void TextureLoadAllTGA()
        {
            TestUtil.AllAssets("*.tga", (filename) => LoaderTGA.LoadFromPath(filename));
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
