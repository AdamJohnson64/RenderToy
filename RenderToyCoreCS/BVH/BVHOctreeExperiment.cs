////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace RenderToy
{
    /// <summary>
    /// Loose Octree BVH (Tasks+Async/Await).
    /// Since threads don't exist in UWP we have to use the task system.
    /// This incomplete model attempts to use some of those systems to build a BVH.
    /// </summary>
    public static partial class BVH
    {
        public static MeshBVH.Node CreateLooseOctreeTASK(Triangle3D[] triangles)
        {
            return CreateLooseOctreeTASK(triangles, MAXIMUM_BVH_DEPTH);
        }
        public static MeshBVH.Node CreateLooseOctreeTASK(Triangle3D[] triangles, int level)
        {
            var result = CreateClippedNode(triangles).Result;
            return result;
        }
        static async Task<MeshBVH.Node> CreateClippedNode(Triangle3D[] triangles)
        {
            Bound3D bound = ComputeBounds(triangles);
            var newchildren = await CreateChildren(triangles, bound);
            if (newchildren.Length != 0) triangles = null;
            return new MeshBVH.Node(bound, triangles, newchildren);
        }
        static async Task<MeshBVH.Node> CreateUnclippedNode(Triangle3D[] triangles, Bound3D bound)
        {
            if (triangles == null) return null;
            var newtriangles = triangles.Where(t => ShapeIntersects(bound, new Triangle3D(t.P0, t.P1, t.P2))).ToArray();
            if (newtriangles.Length == 0 || newtriangles.Length < 16 || triangles.Length == newtriangles.Length) return null;
            return await CreateClippedNode(newtriangles);
        }
        static async Task<MeshBVH.Node[]> CreateChildren(Triangle3D[] triangles, Bound3D bound)
        {
            var children = EnumerateSplit222(bound)
                .Select(newbound => CreateUnclippedNode(triangles, newbound))
                .ToArray();
            var childnodes = await Task.WhenAll(children);
            return childnodes
                .Where(node => node != null)
                .ToArray();
        }
    }
    /// <summary>
    /// Loose Octree BVH (non-recursive).
    /// This model uses a continuously expanded stack of nodes updated in a loop.
    /// It might be possible to exploit more parallelism here later.
    /// </summary>
    public static partial class BVH
    {
        public static MeshBVH.Node CreateLooseOctreeST(Triangle3D[] triangles)
        {
            return CreateLooseOctreeST(triangles, MAXIMUM_BVH_DEPTH);
        }
        public static MeshBVH.Node CreateLooseOctreeST(Triangle3D[] triangles, int level)
        {
            var openlist = new List<WorkingNode>();
            // Add the root node to the queue.
            openlist.Add(new WorkingNode { ParentIndex = -1, Bound = ComputeBounds(triangles), Triangles = triangles });
            // Start at the top of the list and start consuming working nodes.
            int openindex = 0;
            while (openindex < openlist.Count)
            {
                var open = openlist[openindex];
                var opennext = EnumerateSplit222(open.Bound)
                    .AsParallel()
                    .Select(bound => GenerateWorkingNode(openindex, open.Triangles, bound))
                    .Where(node => node != null)
                    .Where(node => node.Triangles.Length > 16)
                    .Where(node => node.Triangles.Length != open.Triangles.Length)
                    .ToArray();
                ++openindex;
                if (opennext.Select(x => x.Triangles.Length).Sum() > open.Triangles.Length * 2) continue;
                if (opennext.Length > 0) open.Triangles = null;
                openlist.AddRange(opennext);
            }
            throw new NotImplementedException();
        }
        [DebuggerDisplay("{ParentIndex} {Triangles.Length}")]
        class WorkingNode
        {
            public int ParentIndex { get; set; }
            public Triangle3D[] Triangles { get; set; }
            public Bound3D Bound { get; set; }
        }
        static WorkingNode GenerateWorkingNode(int parentindex, Triangle3D[] oldtriangles, Bound3D newbound)
        {
            var newnode = new WorkingNode();
            newnode.ParentIndex = parentindex;
            newnode.Triangles = oldtriangles.Where(t => ShapeIntersects(newbound, new Triangle3D(t.P0, t.P1, t.P2))).ToArray();
            if (newnode.Triangles.Length == 0) return null;
            newnode.Bound = ComputeBounds(newnode.Triangles);
            return newnode;
        }
    }
    /// <summary>
    /// Loose Octree BVH (Multithread).
    /// Work in progress.
    /// </summary>
    public static partial class BVH
    {
        public static MeshBVH.Node CreateLooseOctreeMT(Triangle3D[] triangles, int level)
        {
            var openlist = new ConcurrentQueue<WorkingNode>();
            var work = new WorkQueue();
            // Add the root node to the queue.
            Action<WorkingNode> processnode = null;
            processnode = (open) =>
            {
                openlist.Enqueue(open);
                var opennext = EnumerateSplit222(open.Bound)
                    .Select(bound => GenerateWorkingNode(-1, open.Triangles, bound))
                    .Where(node => node != null)
                    .Where(node => node.Triangles.Length > 16)
                    .Where(node => node.Triangles.Length != open.Triangles.Length)
                    .ToArray();
                foreach (var child in opennext)
                {
                    work.Queue(() => processnode(child));
                }
            };
            work.Queue(() => processnode(new WorkingNode { ParentIndex = -1, Bound = ComputeBounds(triangles), Triangles = triangles }));
            work.Start();
            // Start at the top of the list and start consuming working nodes.
            throw new NotImplementedException();
        }
    }
    /// <summary>
    /// Loose Octree BVH (Multithread).
    /// Work in progress.
    /// </summary>
    public static partial class BVH
    {
        class WorkingNode2
        {
            public int ParentIndex { get; set; }
            public Bound3D Bound { get; set; }
            public Triangle3D[] TrianglesPre { get; set; }
            public Triangle3D[] Triangles { get; set; }
        }
        public static MeshBVH.Node CreateLooseOctreeMT2(Triangle3D[] triangles, int level)
        {
            var openlist = new ConcurrentQueue<WorkingNode2>();
            var work = new WorkQueue();
            // Add the root node to the queue.
            Action<WorkingNode2> processnode = null;
            processnode = (open) =>
            {
                if (open.TrianglesPre == null && open.Triangles != null)
                {
                    var opennext = EnumerateSplit222(open.Bound)
                        .Select(bound => new WorkingNode2 { ParentIndex = -1, Bound = bound, TrianglesPre = open.Triangles })
                        .ToArray();
                    //openlist.Enqueue(open);
                    foreach (var child in opennext)
                    {
                        work.Queue(() => processnode(child));
                    }
                }
                if (open.TrianglesPre != null && open.Triangles == null)
                {
                    open.Triangles = open.TrianglesPre.Where(t => ShapeIntersects(open.Bound, new Triangle3D(t.P0, t.P1, t.P2))).ToArray();
                    if (open.Triangles.Length < 16) return;
                    open.TrianglesPre = null;
                    open.Bound = ComputeBounds(open.Triangles);
                    //openlist.Enqueue(open);
                    var opennext = EnumerateSplit222(open.Bound)
                        .Select(bound => new WorkingNode2 { ParentIndex = -1, Bound = bound, TrianglesPre = open.Triangles })
                        .ToArray();
                    foreach (var child in opennext)
                    {
                        work.Queue(() => processnode(child));
                    }
                }
            };
            //processnode(new WorkingNode { ParentIndex = -1, Bound = ComputeBounds(triangles), Triangles = triangles });
            work.Queue(() => processnode(new WorkingNode2 { ParentIndex = -1, Bound = ComputeBounds(triangles), Triangles = triangles }));
            work.Start();
            // Start at the top of the list and start consuming working nodes.
            throw new NotImplementedException();
        }
    }
}