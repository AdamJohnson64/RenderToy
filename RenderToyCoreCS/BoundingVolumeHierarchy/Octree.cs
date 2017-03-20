////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph.Meshes;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.BoundingVolumeHierarchy
{
    public static partial class Octree
    {
        public static MeshBVH Create(Triangle3D[] triangles)
        {
            return Create(triangles, CommonBVH.MAXIMUM_BVH_DEPTH);
        }
        static MeshBVH Create(Triangle3D[] triangles, int level)
        {
            if (triangles == null || triangles.Length == 0) return null;
            Bound3D bound = CommonBVH.ComputeBounds(triangles);
            if (level <= 0) goto EMITUNMODIFIED;
            if (triangles.Length <= 4) goto EMITUNMODIFIED;
            double mid_x = (bound.Min.X + bound.Max.X) / 2;
            double mid_y = (bound.Min.Y + bound.Max.Y) / 2;
            double mid_z = (bound.Min.Z + bound.Max.Z) / 2;
            List<Triangle3D>[] children_triangles = {
                new List<Triangle3D>(), // -X, -Y, -Z
                new List<Triangle3D>(), // +X, -Y, -Z
                new List<Triangle3D>(), // -X, +Y, -Z
                new List<Triangle3D>(), // +X, +Y, -Z
                new List<Triangle3D>(), // -X, -Y, +Z
                new List<Triangle3D>(), // +X, -Y, +Z
                new List<Triangle3D>(), // -X, +Y, +Z
                new List<Triangle3D>()  // +X, +Y, +Z
            };
            foreach (var triangle in triangles)
            {
                byte triangle_in_child_mask = 0xFF; // Initially in all nodes
                double distance_to_plane;
                {
                    bool negative_x = false, positive_x = false;
                    distance_to_plane = triangle.P0.X - mid_x; negative_x |= distance_to_plane <= 0; positive_x |= distance_to_plane >= 0;
                    distance_to_plane = triangle.P1.X - mid_x; negative_x |= distance_to_plane <= 0; positive_x |= distance_to_plane >= 0;
                    distance_to_plane = triangle.P2.X - mid_x; negative_x |= distance_to_plane <= 0; positive_x |= distance_to_plane >= 0;
                    byte triangle_sides_mask = 0;
                    if (negative_x) triangle_sides_mask |= 0x55; // 01010101b, all children on the negative x side of this cube
                    if (positive_x) triangle_sides_mask |= 0xAA; // 10101010b, all children on the positive x side of this cube
                    triangle_in_child_mask &= triangle_sides_mask;
                }
                {
                    bool negative_y = false, positive_y = false;
                    distance_to_plane = triangle.P0.Y - mid_y; negative_y |= distance_to_plane <= 0; positive_y |= distance_to_plane >= 0;
                    distance_to_plane = triangle.P1.Y - mid_y; negative_y |= distance_to_plane <= 0; positive_y |= distance_to_plane >= 0;
                    distance_to_plane = triangle.P2.Y - mid_y; negative_y |= distance_to_plane <= 0; positive_y |= distance_to_plane >= 0;
                    byte triangle_sides_mask = 0;
                    if (negative_y) triangle_sides_mask |= 0x33; // 00110011b, all children on the negative y side of this cube
                    if (positive_y) triangle_sides_mask |= 0xCC; // 11001100b, all children on the positive y side of this cube
                    triangle_in_child_mask &= triangle_sides_mask;
                }
                {
                    bool negative_z = false, positive_z = false;
                    distance_to_plane = triangle.P0.Z - mid_z; negative_z |= distance_to_plane <= 0; positive_z |= distance_to_plane >= 0;
                    distance_to_plane = triangle.P1.Z - mid_z; negative_z |= distance_to_plane <= 0; positive_z |= distance_to_plane >= 0;
                    distance_to_plane = triangle.P2.Z - mid_z; negative_z |= distance_to_plane <= 0; positive_z |= distance_to_plane >= 0;
                    byte triangle_sides_mask = 0;
                    if (negative_z) triangle_sides_mask |= 0x0F; // 00001111b, all children on the negative z side of this cube
                    if (positive_z) triangle_sides_mask |= 0xF0; // 11110000b, all children on the positive z side of this cube
                    triangle_in_child_mask &= triangle_sides_mask;
                }
                if ((triangle_in_child_mask & 0x01) == 0x01) children_triangles[0].Add(triangle);
                if ((triangle_in_child_mask & 0x02) == 0x02) children_triangles[1].Add(triangle);
                if ((triangle_in_child_mask & 0x04) == 0x04) children_triangles[2].Add(triangle);
                if ((triangle_in_child_mask & 0x08) == 0x08) children_triangles[3].Add(triangle);
                if ((triangle_in_child_mask & 0x10) == 0x10) children_triangles[4].Add(triangle);
                if ((triangle_in_child_mask & 0x20) == 0x20) children_triangles[5].Add(triangle);
                if ((triangle_in_child_mask & 0x40) == 0x40) children_triangles[6].Add(triangle);
                if ((triangle_in_child_mask & 0x80) == 0x80) children_triangles[7].Add(triangle);
            }
            // If all triangles ended up in one node then abort; it won't get any better.
            if (children_triangles.Any(x => x.Count == triangles.Length)) goto EMITUNMODIFIED;
            return new MeshBVH(bound, null, children_triangles.Select(x => Create(x.ToArray(), level - 1)).Where(x => x != null).ToArray());
            EMITUNMODIFIED:
            return new MeshBVH(bound, triangles, null);
        }
    }
}