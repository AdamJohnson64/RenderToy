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
    public class TransformedObject
    {
        TransformedObject(IMaterial nodematerial, IPrimitive nodeprimitive, ITransform nodetransform, Vector4D nodewirecolor, Matrix3D transformparent, Matrix3D transform)
        {
            TransformParent = transformparent;
            Transform = transform;
            NodeMaterial = nodematerial;
            NodePrimitive = nodeprimitive;
            NodeTransform = nodetransform;
            NodeWireColor = nodewirecolor;
        }
        public readonly Matrix3D TransformParent;
        public readonly Matrix3D Transform;
        public readonly IMaterial NodeMaterial;
        public readonly IPrimitive NodePrimitive;
        public readonly ITransform NodeTransform;
        public readonly Vector4D NodeWireColor;
        public static IEnumerable<TransformedObject> Enumerate(IScene scene)
        {
            if (scene == null) yield break;
            foreach (INode root in scene.Children)
            {
                foreach (TransformedObject tobj in Enumerate(root, Matrix3D.Identity))
                {
                    yield return tobj;
                }
            }
        }
        static IEnumerable<TransformedObject> Enumerate(INode node, Matrix3D parenttransform)
        {
            Matrix3D localtransform = parenttransform * node.Transform.Transform;
            yield return new TransformedObject(node.Material, node.Primitive, node.Transform, node.WireColor, parenttransform, localtransform);
            foreach (INode child in node.Children)
            {
                foreach (TransformedObject transformedchild in Enumerate(child, localtransform))
                {
                    yield return transformedchild;
                }
            }
        }
    }
}