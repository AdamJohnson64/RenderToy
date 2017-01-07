using System;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public class GeometrySwatch : FrameworkElement
    {
        public GeometrySwatch(object primitive)
        {
            Width = 32;
            Height = 32;
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
            scene.AddChild(new Node(new TransformMatrix3D(Matrix3D.Identity), Primitive, Colors.DarkGray));
            Matrix3D View = MathHelp.CreateTranslateMatrix(0, 0, 3);
            Matrix3D ProjectionWindow = CameraPerspective.CreateProjection(0.001, 100, 45.0 * Math.PI / 180.0, 45.0 * Math.PI / 180.0);
            Matrix3D mvp = View * ProjectionWindow;
            // Prefer to render the primitive as a raytraced object first.
            if (Primitive is IRayTest)
            {
                ControlUtil.RenderRaytrace(drawingContext, ActualWidth, ActualHeight, scene, mvp, 64, 64);
            }
            // Then try a parametric wireframe.
            if (Primitive is IParametricUV)
            {
                ControlUtil.RenderWireframeWPF(drawingContext, ActualWidth, ActualHeight, scene, mvp);
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