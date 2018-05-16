////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;

namespace RenderToy.Transforms
{
    public class TransformQuaternion : ITransform
    {
        public TransformQuaternion()
        {
        }
        public TransformQuaternion(Vector3D position)
        {
            Position = position;
        }
        public Matrix3D Transform
        {
            get
            {
                Matrix3D rotate = MathHelp.CreateMatrixRotation(Rotation);
                Matrix3D translate = MathHelp.CreateMatrixTranslate(Position.X, Position.Y, Position.Z);
                Matrix3D result = rotate * translate;
                return result;
            }
            set
            {
                this.Rotation = MathHelp.CreateQuaternionRotation(value);
                this.Position = new Vector3D(value.M41, value.M42, value.M43);
            }
        }
        public void TranslatePost(Vector3D offset)
        {
            Matrix3D frame = MathHelp.CreateMatrixRotation(Rotation);
            Position += MathHelp.TransformPoint(frame, offset);
        }
        public void RotatePost(Quaternion rotate)
        {
            Rotation = Rotation * rotate;
        }
        public void RotatePre(Quaternion rotate)
        {
            Rotation = rotate * Rotation;
        }
        Vector3D Position = new Vector3D(0, 0, 0);
        Quaternion Rotation = new Quaternion(0, 0, 0, 1);
    }
}