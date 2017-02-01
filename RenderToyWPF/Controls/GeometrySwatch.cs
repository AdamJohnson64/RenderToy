////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace RenderToy
{
    public class GeometrySwatch : FrameworkElement
    {
        public GeometrySwatch(object primitive)
        {
            Width = 64;
            Height = 64;
            Primitive = primitive;
        }
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            DragDrop.DoDragDrop(this, new Sphere(), DragDropEffects.Copy);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            Scene scene = new Scene();
            scene.AddChild(new Node(new TransformMatrix3D(Matrix3D.Identity), Primitive, Materials.DarkGray, Materials.PlasticRed));
            Matrix3D Camera = MathHelp.CreateMatrixLookAt(new Point3D(2, 2, -2), new Point3D(0, 0, 0), new Vector3D(0, 1, 0));
            Matrix3D View = MathHelp.Invert(Camera);
            Matrix3D Projection = CameraPerspective.CreateProjection(0.001, 100, 45.0 * Math.PI / 180.0, 45.0 * Math.PI / 180.0);
            Matrix3D mvp = View * Projection;
            if (Primitive is IParametricUV)
            {
                drawingContext.DrawImage(ImageHelp.CreateImage(Render.Wireframe, scene, mvp, 64, 64), new Rect(0, 0, ActualWidth, ActualHeight));
                return;
            }
        }
        private object Primitive;
    }
    public class GeometrySwatchBezierPatch : GeometrySwatch
    {
        public GeometrySwatchBezierPatch() : base(new BezierPatch()) { }
    }
    public class GeometrySwatchCylinder : GeometrySwatch
    {
        public GeometrySwatchCylinder() : base(new Cylinder()) { }
    }
    public class GeometrySwatchSphere : GeometrySwatch
    {
        public GeometrySwatchSphere() : base(new Sphere()) { }
    }
}