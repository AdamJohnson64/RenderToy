////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy
{
    public interface ITransformed
    {
        Matrix3D Transform { get; }
    }
    public class TransformMatrix3D : ITransformed
    {
        public TransformMatrix3D(Matrix3D value)
        {
            this.value = value;
        }
        public Matrix3D Transform
        {
            get
            {
                return value;
            }
        }
        Matrix3D value;
    }
    public class TransformPosQuat : ITransformed
    {
        public Matrix3D Transform
        {
            get
            {
                Matrix3D rotate = MathHelp.CreateMatrixRotation(Rotation);
                Matrix3D translate = MathHelp.CreateMatrixTranslate(Position.X, Position.Y, Position.Z);
                Matrix3D result = rotate * translate;
                return result;
            }
        }
        public void TranslatePost(Vector3D offset)
        {
            Matrix3D frame = MathHelp.CreateMatrixRotation(Rotation);
            Position += frame.Transform(offset);
        }
        public void RotatePost(Quaternion rotate)
        {
            Rotation = Rotation * rotate;
        }
        public void RotatePre(Quaternion rotate)
        {
            Rotation = rotate * Rotation;
        }
        public Point3D Position = new Point3D(0, 0, 0);
        public Quaternion Rotation = new Quaternion(0, 0, 0, 1);
    }
}