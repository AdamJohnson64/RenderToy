////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace RenderToy
{
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
}