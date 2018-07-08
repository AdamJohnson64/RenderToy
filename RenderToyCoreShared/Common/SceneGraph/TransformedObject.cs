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
        static Matrix3D ConvertMatrix(float[] matrix, float scale)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix[0];
            m.M21 = matrix[1];
            m.M31 = -matrix[2];
            m.M41 = matrix[3] * scale;
            m.M12 = matrix[4];
            m.M22 = matrix[5];
            m.M32 = -matrix[6];
            m.M42 = matrix[7] * scale;
            m.M13 = matrix[8];
            m.M23 = matrix[9];
            m.M33 = -matrix[10];
            m.M43 = matrix[11] * -scale;
            m.M44 = 1;
            return m;
        }
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
#if OPENVR_INSTALLED
            float scale = 10.0f;
            {
                var matin = new float[4 * 3];
                if (OpenVR.LocateDeviceId(matin, 0))
                {
                    var matout = ConvertMatrix(matin, scale);
                    yield return new TransformedObject(new Node("Head", new TransformMatrix(matout), VRTESTPRIMITIVE, StockMaterials.Green, StockMaterials.PlasticGreen), matout);
                }
            }
            {
                var matin = new float[4 * 3];
                if (OpenVR.LocateDeviceRole(matin, TrackedControllerRole.RightHand))
                {
                    var matout = ConvertMatrix(matin, scale);
                    yield return new TransformedObject(new Node("Right Hand", new TransformMatrix(matout), VRTESTPRIMITIVE, StockMaterials.Green, StockMaterials.PlasticGreen), matout);
                }
            }
            {
                var matin = new float[4 * 3];
                if (OpenVR.LocateDeviceRole(matin, TrackedControllerRole.LeftHand))
                {
                    var matout = ConvertMatrix(matin, scale);
                    yield return new TransformedObject(new Node("Left Hand", new TransformMatrix(matout), VRTESTPRIMITIVE, StockMaterials.Red, StockMaterials.PlasticRed), matout);
                }
            }
#endif // OPENVR_INSTALLED
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