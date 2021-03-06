﻿using RenderToy.Math;

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