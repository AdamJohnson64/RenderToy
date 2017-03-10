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
    /// Loose Octree BVH (naive recursive reference model).
    /// There's some parallelism expressed in here but it's very weak and not well thought out.
    /// Other models can do much better, but this works for now.
    /// </summary>
    public static partial class BVH
    {
        public static MeshBVH.Node CreateLooseOctree(Triangle3D[] triangles, int level)
        {
            Bound3D bound = ComputeBounds(triangles);
            // Stop at 8 levels.
            if (level <= 0) goto EMITUNMODIFIED;
            // Stop at 4 triangles
            if (triangles.Length < 4) goto EMITUNMODIFIED;
            // Slice this region into 8 subcubes (roughly octree).
            var children = EnumerateSplit222(bound)
                .Select(box => CreateLooseOctreeChild(triangles, level, box))
                .Where(node => node != null)
                .ToArray();
            // Form the new node.
            var newnode = new MeshBVH.Node(bound, null, children);
            // If this split blows up the number of triangles significantly then reject it.
            var numtriangles = MeshBVH
                .EnumerateNodes(newnode)
                .AsParallel()
                .Where(n => n.Triangles != null)
                .SelectMany(n => n.Triangles)
                .Count();
            if (numtriangles > triangles.Count() * 2) goto EMITUNMODIFIED;
            // Otherwise return the new node.
            return newnode;
        EMITUNMODIFIED:
            return new MeshBVH.Node(bound, triangles, null);
        }
        static MeshBVH.Node CreateLooseOctreeChild(Triangle3D[] triangles, int level, Bound3D box)
        {
            // Partition the triangles.
            Triangle3D[] contained_triangles = CreateTriangleList(triangles, box);
            // If there are no triangles in this child node then skip it entirely.
            if (contained_triangles.Length == 0) return null;
            // If all the triangles are in this node then skip it.
            if (contained_triangles.Length == triangles.Length) return null;
            // Generate the new child node.
            // Also, recompute the extents of this bounding volume.
            // It's possible if the mesh has large amounts of space crossing the clip plane such that the bounds are now too big.
            return CreateLooseOctree(contained_triangles, level - 1);
        }
        static Triangle3D[] CreateTriangleList(Triangle3D[] triangles, Bound3D box)
        {
            if (triangles.Length < 32)
            {
                return triangles.Where(t => ShapeIntersects(box, new Triangle3D(t.P0, t.P1, t.P2))).ToArray();
            }
            else
            {
                return triangles.AsParallel().Where(t => ShapeIntersects(box, new Triangle3D(t.P0, t.P1, t.P2))).ToArray();
            }
        }
        static IEnumerable<Bound3D> EnumerateSplit222(Bound3D box)
        {
            double midx = (box.Min.X + box.Max.X) / 2;
            double midy = (box.Min.Y + box.Max.Y) / 2;
            double midz = (box.Min.Z + box.Max.Z) / 2;
            for (int z = 0; z < 2; ++z)
            {
                for (int y = 0; y < 2; ++y)
                {
                    for (int x = 0; x < 2; ++x)
                    {
                        // Compute the expected subcube extents.
                        Vector3D min = new Vector3D(x == 0 ? box.Min.X : midx, y == 0 ? box.Min.Y : midy, z == 0 ? box.Min.Z : midz);
                        Vector3D max = new Vector3D(x == 0 ? midx : box.Max.X, y == 0 ? midy : box.Max.Y, z == 0 ? midz : box.Max.Z);
                        yield return new Bound3D(min, max);
                    }
                }
            }
        }
    }
    /// <summary>
    /// Loose Octree BVH (Tasks+Async/Await).
    /// Since threads don't exist in UWP we have to use the task system.
    /// This incomplete model attempts to use some of those systems to build a BVH.
    /// </summary>
    public static partial class BVH
    {
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
        [DebuggerDisplayAttribute("{ParentIndex} {Triangles.Length}")]
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