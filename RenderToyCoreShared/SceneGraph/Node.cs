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
    /// <summary>
    /// A generic node in a scene consisting of a transformed primitive with a given material.
    /// A node is also a type of scene given that it contains children.
    /// This property allows the passing of children as scene subgraphs for partial evaluation.
    /// </summary>
    public interface INode : IScene
    {
        string GetName();
        ITransform GetTransform();
        IPrimitive GetPrimitive();
        Vector4D GetWireColor();
        IMaterial GetMaterial();
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
        public string GetName() { return name; }
        public ITransform GetTransform() { return transform; }
        public IPrimitive GetPrimitive() { return primitive; }
        public Vector4D GetWireColor() { return wirecolor; }
        public IMaterial GetMaterial() { return material; }
        public IReadOnlyList<INode> GetChildren() { return children; }
        readonly string name;
        readonly ITransform transform;
        readonly IPrimitive primitive;
        readonly Vector4D wirecolor;
        readonly IMaterial material;
        readonly List<INode> children = new List<INode>();
    }
}