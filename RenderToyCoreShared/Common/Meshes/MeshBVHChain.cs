////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.BoundingVolumeHierarchy;
using System.Collections.Generic;

namespace RenderToy.Meshes
{
    /// <summary>
    /// The MeshBVHChain builder is similar to a standard BVH with child arrays
    /// with the exception that we encode a linked chain of nodes that runs
    /// through all children and siblings. Using this structure allows us to
    /// walk the node tree iteratively instead of recursively which is far more
    /// GPU friendly.
    /// </summary>
    public class MeshBVHChain : MeshBVH
    {
        public static MeshBVHChain Create(MeshBVH root)
        {
            var result = Create(root, null);
            Chain(result);
            return result;
        }
        static MeshBVHChain Create(MeshBVH root, MeshBVHChain parent)
        {
            var thisroot = new MeshBVHChain(root.Bound, root.Triangles, parent);
            if (root.Children != null)
            {
                var nextroots = new List<MeshBVHChain>();
                MeshBVHChain previous = null;
                for (int i = 0; i < root.Children.Length; ++i)
                {
                    var next = Create(root.Children[i], thisroot);
                    if (previous != null)
                    {
                        previous.Sibling = next;
                    }
                    nextroots.Add(next);
                    previous = next;
                }
                thisroot.Children = nextroots.ToArray();
                thisroot.Child = nextroots.Count > 0 ? nextroots[0] : null;
            }
            return thisroot;
        }
        static void Chain(MeshBVHChain root)
        {
            var walk = root;
            while (walk != null)
            {
                if (walk.Sibling == null && walk.Parent != null)
                {
                    walk.Sibling = walk.Parent.Sibling;
                }
                if (walk.Child != null)
                {
                    walk = walk.Child;
                }
                else
                {
                    walk = walk.Sibling;
                }
            }
        }
        MeshBVHChain(Bound3D bound, Triangle3D[] triangles, MeshBVHChain parent)
            : base(bound, triangles, null)
        {
            Parent = parent;
        }
        public MeshBVHChain Child { get; protected set; }
        public MeshBVHChain Sibling { get; protected set; }
        public MeshBVHChain Parent { get; protected set; }
    }
}