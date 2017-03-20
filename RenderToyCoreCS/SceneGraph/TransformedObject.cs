////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;

namespace RenderToy.SceneGraph
{
    public class TransformedObject
    {
        TransformedObject(Node node, Matrix3D transform)
        {
            Node = node;
            Transform = transform;
        }
        public readonly Node Node;
        public readonly Matrix3D Transform;
        public static IEnumerable<TransformedObject> Enumerate(Scene scene)
        {
            if (scene == null) yield break;
            foreach (Node root in scene.Children)
            {
                foreach (TransformedObject tobj in Enumerate(root, Matrix3D.Identity))
                {
                    yield return tobj;
                }
            }
        }
        static IEnumerable<TransformedObject> Enumerate(Node node, Matrix3D parenttransform)
        {
            Matrix3D localtransform = parenttransform * node.Transform.Transform;
            yield return new TransformedObject(node, localtransform);
            foreach (Node child in node.Children)
            {
                foreach (TransformedObject transformedchild in Enumerate(child, localtransform))
                {
                    yield return transformedchild;
                }
            }
        }
    }
}