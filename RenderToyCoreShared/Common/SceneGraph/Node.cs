////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Primitives;
using RenderToy.Transforms;
using System.Collections.Generic;

namespace RenderToy.SceneGraph
{
    /// <summary>
    /// A generic node in a scene consisting of a transformed primitive with a given material.
    /// A node is also a type of scene given that it contains children.
    /// This property allows the passing of children as scene subgraphs for partial evaluation.
    /// </summary>
    public interface INode : IScene
    {
        string Name { get; }
        ITransform Transform { get; }
        IPrimitive Primitive { get; }
        Vector4D WireColor { get; }
        IMaterial Material { get; }
    }
    public class Node : INode
    {
        public Node(string name, ITransform transform, IPrimitive primitive, Vector4D wirecolor, IMaterial material)
        {
            this.name = name;
            this.transform = transform;
            this.primitive = primitive;
            this.wirecolor = wirecolor;
            this.material = material;
        }
        public string Name { get { return name; } }
        public ITransform Transform { get { return transform; } }
        public IPrimitive Primitive { get { return primitive; } }
        public Vector4D WireColor { get { return wirecolor; } }
        public IMaterial Material { get { return material; } }
        public IReadOnlyList<INode> Children { get { return children; } }
        readonly string name;
        readonly ITransform transform;
        readonly IPrimitive primitive;
        readonly Vector4D wirecolor;
        readonly IMaterial material;
        public readonly List<INode> children = new List<INode>();
    }
}