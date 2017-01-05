using System.Windows.Media.Media3D;

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
                return MathHelp.ToMatrix(Rotation) * MathHelp.ToMatrix(Position);
            }
        }
        public void TranslatePost(Vector3D offset)
        {
            Matrix3D frame = MathHelp.ToMatrix(Rotation);
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
        public Vector3D Position = new Vector3D(0, 0, 0);
        public Quaternion Rotation = new Quaternion(0, 0, 0, 1);
    }
}