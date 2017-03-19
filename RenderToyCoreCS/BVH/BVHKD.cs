////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;

namespace RenderToy
{
    public static partial class BVH
    {
        public static MeshBVH CreateKD(Triangle3D[] triangles)
        {
            return CreateKD(triangles, MAXIMUM_BVH_DEPTH);
        }
        public static MeshBVH CreateKD(Triangle3D[] triangles, int level)
        {
            var bound = ComputeBounds(triangles);
            if (level <= 0) goto EMITUNMODIFIED;
            if (triangles.Length <= 16) goto EMITUNMODIFIED;
            // Select an axis to cut.
            // For now we'll choose the longest axis of the AABB, X/Y/Z.
            CutAxis axis = CutAxis.X;
            double dx = bound.Max.X - bound.Min.X;
            double dy = bound.Max.Y - bound.Min.Y;
            double dz = bound.Max.Z - bound.Min.Z;
            if (dx > dy && dx > dz) axis = CutAxis.X;
            if (dy > dx && dy > dz) axis = CutAxis.Y;
            if (dz > dx && dz > dy) axis = CutAxis.Z;
            // Now determine a cut plane on this axis to form two new boxes.
            // For now we'll perform a median cut which is sub-optimal but enough to prove a point.
            Vector3D plane_normal;
            double plane_distance;
            switch (axis)
            {
                case CutAxis.X:
                    plane_normal = new Vector3D(1, 0, 0);
                    plane_distance = (bound.Min.X + bound.Max.X) / 2;
                    break;
                case CutAxis.Y:
                    plane_normal = new Vector3D(0, 1, 0);
                    plane_distance = (bound.Min.Y + bound.Max.Y) / 2;
                    break;
                case CutAxis.Z:
                    plane_normal = new Vector3D(0, 0, 1);
                    plane_distance = (bound.Min.Z + bound.Max.Z) / 2;
                    break;
                default:
                    throw new NotImplementedException();
            }
            // Partition the triangles into back/front sets, triangles crossing the plane should appear in both sets.
            var triangles_back = new List<Triangle3D>(triangles.Length);
            var triangles_frnt = new List<Triangle3D>(triangles.Length);
            foreach (var t in triangles)
            {
                double side0 = MathHelp.Dot(t.P0, plane_normal) - plane_distance;
                double side1 = MathHelp.Dot(t.P1, plane_normal) - plane_distance;
                double side2 = MathHelp.Dot(t.P2, plane_normal) - plane_distance;
                if (side0 <= 0 || side1 <= 0 || side2 <= 0)
                {
                    triangles_back.Add(t);
                }
                if (side0 >= 0 || side1 >= 0 || side2 >= 0)
                {
                    triangles_frnt.Add(t);
                }
            }
            triangles_back.TrimExcess();
            triangles_frnt.TrimExcess();
            // If everything is in either set then don't bother clipping this node.
            if (triangles_back.Count == triangles.Length || triangles_frnt.Count == triangles.Length) goto EMITUNMODIFIED;
            // If we amplify the triangle count by more than 50% then don't bother clipping this node.
            if (triangles_back.Count + triangles_frnt.Count > triangles.Length * 3 / 2) goto EMITUNMODIFIED;
            // Otherwise we have a good node pair.
            var node_back = CreateKD(triangles_back.ToArray(), level - 1);
            var node_frnt = CreateKD(triangles_frnt.ToArray(), level - 1);
            return new MeshBVH(bound, null, new[] { node_back, node_frnt });
        EMITUNMODIFIED:
            return new MeshBVH(bound, triangles, null);
        }
    }
    enum CutAxis { X, Y, Z }
}