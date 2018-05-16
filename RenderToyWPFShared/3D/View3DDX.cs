using RenderToy.PipelineModel;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public class View3DDX : System.Windows.Controls.Image
    {
        public static DependencyProperty SceneProperty = DependencyProperty.Register("Scene", typeof(IScene), typeof(View3DDX), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));
        public IScene Scene
        {
            get { return (IScene)GetValue(SceneProperty); }
            set { SetValue(SceneProperty, value); }
        }
        public static DependencyProperty CameraProperty = DependencyProperty.Register("Camera", typeof(Camera), typeof(View3DDX), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));
        public Camera Camera
        {
            get { return (Camera)GetValue(CameraProperty); }
            set { SetValue(CameraProperty, value); }
        }
        public View3DDX()
        {
            d3dimage = new D3DImage();
            rendertarget = device.CreateRenderTarget(256, 256, D3DFormat.A8R8G8B8, D3DMultisample.None, 0, 1);
            d3dimage.Lock();
            d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, rendertarget.ManagedPtr);
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, 256, 256));
            d3dimage.Unlock();
            Source = d3dimage;
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            device.SetRenderTarget(0, rendertarget);
            device.BeginScene();
            device.Clear(D3DClear.Target, 0xFF0000FF, 1.0f, 0);
            device.SetFVF(D3DFvf.XYZ | D3DFvf.Diffuse);
            device.SetRenderState(D3DRenderState.ZEnable, 1U);
            device.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
            device.SetRenderState(D3DRenderState.Lighting, 0);
            if (Scene != null && Camera != null)
            {
                var mvp = (Matrix3D)Camera.GetValue(Camera.TransformProperty);
                foreach (var transformedobject in TransformedObject.Enumerate(Scene))
                {
                    var A = PrimitiveAssembly.CreateTriangles(transformedobject.Node.GetPrimitive());
                    var B = A.Select(i => new RenderD3D.XYZDiffuse { X = (float)i.X, Y = (float)i.Y, Z = (float)i.Z, Diffuse = Rasterization.ColorToUInt32(transformedobject.Node.GetWireColor()) });
                    var vertexbuffer = B.ToArray();
                    device.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(D3DMatrix.Convert(transformedobject.Transform * mvp), 0));
                    device.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.TriangleList, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(RenderD3D.XYZDiffuse)));
                }
            }
            device.EndScene();
            base.OnRender(drawingContext);
        }
        D3DImage d3dimage;
        Direct3DSurface9 rendertarget;
        static Direct3D9 d3d = new Direct3D9();
        static Direct3DDevice9 device = d3d.CreateDevice();
    }
}