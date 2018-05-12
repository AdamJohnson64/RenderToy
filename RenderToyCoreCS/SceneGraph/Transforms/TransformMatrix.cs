////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;

namespace RenderToy.Transforms
{
    public class TransformMatrix : ITransform
    {
        public TransformMatrix(Matrix3D value)
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
}