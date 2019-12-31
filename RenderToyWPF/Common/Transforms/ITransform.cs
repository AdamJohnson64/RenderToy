using RenderToy.Math;

namespace RenderToy.Transforms
{
    public interface ITransform
    {
        Matrix3D Transform { get; }
    }
}