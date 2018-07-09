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
        TransformedObject(INode node, Matrix3D transform)
        {
            Node = node;
            Transform = transform;
        }
        public readonly INode Node;
        public readonly Matrix3D Transform;
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
#if OPENVR_INSTALLED
        static IPrimitive VRTESTPRIMITIVE = new Sphere();
#endif // OPENVR_INSTALLED
        static IEnumerable<TransformedObject> Enumerate(INode node, Matrix3D parenttransform)
        {
            Matrix3D localtransform = parenttransform * node.Transform.Transform;
            yield return new TransformedObject(node, localtransform);
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