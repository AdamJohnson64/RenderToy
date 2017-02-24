﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;

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
        public TransformPosQuat()
        {
        }
        public TransformPosQuat(Vector3D position)
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
        }
        public void TranslatePost(Vector3D offset)
        {
            Matrix3D frame = MathHelp.CreateMatrixRotation(Rotation);
            Position += MathHelp.TransformPoint(frame, offset);
            InvalidateTransform();
        }
        public void RotatePost(Quaternion rotate)
        {
            Rotation = Rotation * rotate;
            InvalidateTransform();
        }
        public void RotatePre(Quaternion rotate)
        {
            Rotation = rotate * Rotation;
            InvalidateTransform();
        }
        public event Action OnTransformChanged;
        void InvalidateTransform()
        {
            if (OnTransformChanged != null) OnTransformChanged();
        }
        Vector3D Position = new Vector3D(0, 0, 0);
        Quaternion Rotation = new Quaternion(0, 0, 0, 1);
    }
}