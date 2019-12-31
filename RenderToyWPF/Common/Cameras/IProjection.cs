using RenderToy.Math;

namespace RenderToy.Cameras
{
    public interface IProjection
    {
        Matrix3D Projection { get; }
    }
}