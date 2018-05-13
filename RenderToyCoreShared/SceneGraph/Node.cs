////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Primitives;
using RenderToy.Transforms;
using RenderToy.Utility;
using System.Collections.Generic;

namespace RenderToy.SceneGraph
{
    public class Node
    {
        public Node(string name, ITransform transform, IPrimitive primitive, Vector4D wirecolor, IMaterial material)
        {
            Name = name;
            Transform = transform;
            Primitive = primitive;
            WireColor = wirecolor;
            Material = material;
        }
        public readonly string Name;
        public readonly ITransform Transform;
        public readonly IPrimitive Primitive;
        public readonly Vector4D WireColor;
        public readonly IMaterial Material;
        public IReadOnlyList<Node> Children
        {
            get { return children; }
        }
        List<Node> children = new List<Node>();
    }
}