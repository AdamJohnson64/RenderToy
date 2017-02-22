////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.IO;
using System.Linq;

namespace RenderToy
{
    class Program
    {
        static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: RenderToyUWP_Art <path>");
                return -1;
            }
            string cwd = args[0];
            // Create the UWP application resources.
            BitmapRaytrace(48, 48)
                .Frame(2, 2)
                .Save(Path.Combine(cwd, "LockScreenLogo.scale-200.png"));
            var aoc = BitmapAOC(1240 / 2, 600 / 2);
            BitmapRaytrace(1240, 600)
                .Overlay(aoc, new Rectangle(0, 0, 1240, 128))
                .Overlay(aoc, new Rectangle(0, 600 - 64, 1240, 64))
                .Overlay(BitmapWireframe(1240, 600))
                .Frame(8, 2)
                .Title()
                .Save(Path.Combine(cwd, "SplashScreen.scale-200.png"));
            BitmapRaytrace(300, 300)
                .Overlay(BitmapWireframe(300, 300))
                .Frame(4, 2)
                .Save(Path.Combine(cwd, "Square150x150Logo.scale-200.png"));
            BitmapRaytrace(88, 88)
                .Frame(2, 2)
                .Save(Path.Combine(cwd, "Square44x44Logo.scale-200.png"));
            BitmapRaytrace(24, 24)
                .Frame(1, 1)
                .Save(Path.Combine(cwd, "Square44x44Logo.targetsize-24_altform-unplated.png"));
            BitmapRaytrace(50, 50)
                .Frame(2, 2).Save(Path.Combine(cwd, "StoreLogo.png"));
            BitmapRaytrace(620, 300)
                .Overlay(BitmapWireframe(620, 300))
                .Frame(4, 2).Save(Path.Combine(cwd, "Wide310x150Logo.scale-200.png"));
            return 0;
        }
        static Bitmap BitmapAOC(int width, int height)
        {
            Bitmap bitmap = new System.Drawing.Bitmap(width, height, PixelFormat.Format32bppArgb);
            BitmapData bitmapdata = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            try
            {
                RenderToyCLI.AmbientOcclusionCPUF64(SceneFormatter.CreateFlatMemoryF64(DefaultScene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(DefaultMVP * CameraPerspective.AspectCorrectFit(width, height))), bitmapdata.Scan0, width, height, bitmapdata.Stride, 0, 256);
            }
            finally
            {
                bitmap.UnlockBits(bitmapdata);
            }
            return bitmap;
        }
        static Bitmap BitmapRaytrace(int width, int height)
        {
            Bitmap bitmap = new System.Drawing.Bitmap(width, height, PixelFormat.Format32bppArgb);
            BitmapData bitmapdata = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            try
            {
                RenderToyCLI.RaytraceCPUF64AA(SceneFormatter.CreateFlatMemoryF64(DefaultScene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(DefaultMVP * CameraPerspective.AspectCorrectFit(width, height))), bitmapdata.Scan0, width, height, bitmapdata.Stride, 4, 4);
            }
            finally
            {
                bitmap.UnlockBits(bitmapdata);
            }
            return bitmap;
        }
        static Bitmap BitmapWireframe(int width, int height)
        {
            Bitmap bitmap = new System.Drawing.Bitmap(width, height, PixelFormat.Format32bppArgb);
            BitmapData bitmapdata = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            try
            {
                RenderToy.RenderCS.WireframeCPUF64(DefaultScene, DefaultMVP * CameraPerspective.AspectCorrectFit(width, height), bitmapdata.Scan0, width, height, bitmapdata.Stride);
            }
            finally
            {
                bitmap.UnlockBits(bitmapdata);
            }
            return bitmap;
        }
        static Scene DefaultScene
        {
            get
            {
                Scene scene = new Scene();
                scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixIdentity()), new Plane(), Materials.DarkGray, new CheckerboardMaterial(Materials.Black, Materials.White)));
                scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-2, 1, 0)), new Sphere(), Materials.Red, Materials.PlasticRed));
                scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(0, 1, 0)), new Sphere(), Materials.Green, Materials.PlasticGreen));
                scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+2, 1, 0)), new Sphere(), Materials.Blue, Materials.PlasticBlue));
                scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(0, 1, -2)), new Sphere(), Materials.Black, Materials.Glass));
                return scene;
            }
        }
        static Matrix3D DefaultView
        {
            get
            {
                return MathHelp.Invert(MathHelp.CreateMatrixLookAt(new Vector3D(-1, 3, -4), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0)));

            }
        }
        static Matrix3D DefaultProjection
        {
            get
            {
                return CameraPerspective.CreateProjection(0.001, 100.0, 45.0, 45.0);
            }
        }
        static Matrix3D DefaultMVP
        {
            get { return DefaultView * DefaultProjection; }
        }
    }
    static class ImageOperation
    {
        public static Bitmap Frame(this Bitmap baseimage, int frameinset, int framewidth)
        {
            Bitmap finalcomposition = new Bitmap(baseimage.Width, baseimage.Height, PixelFormat.Format32bppArgb);
            using (Graphics g = Graphics.FromImage(finalcomposition))
            {
                // Use best quality rendering.
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
                // Draw in the bitmap and frame it with a shadow.
                g.FillRectangle(new SolidBrush(Color.FromArgb(64, Color.Black)), new Rectangle(frameinset * 2, frameinset * 2, baseimage.Width - frameinset * 2, baseimage.Height - frameinset * 2));
                g.SetClip(new Rectangle(frameinset, frameinset, baseimage.Width - frameinset * 2, baseimage.Height - frameinset * 2));
                g.DrawImage(baseimage, new Point(0, 0));
                g.ResetClip();
                g.DrawRectangle(new Pen(Brushes.LightGray, framewidth), new Rectangle(frameinset, frameinset, baseimage.Width - frameinset * 2, baseimage.Height - frameinset * 2));
            }
            return finalcomposition;
        }
        public static Bitmap Overlay(this Bitmap baseimage, Bitmap addimage)
        {
            return Overlay(baseimage, addimage, new Rectangle(0, 0, baseimage.Width, baseimage.Height));
        }
        public static Bitmap Overlay(this Bitmap baseimage, Bitmap addimage, Rectangle clip)
        {
            Bitmap finalcomposition = new Bitmap(baseimage.Width, baseimage.Height, PixelFormat.Format32bppArgb);
            using (Graphics g = Graphics.FromImage(finalcomposition))
            {
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.DrawImage(baseimage, new Point(0, 0));
                g.SetClip(clip);
                g.DrawImage(addimage, new Rectangle(0, 0, baseimage.Width, baseimage.Height));
                g.ResetClip();
            }
            return finalcomposition;
        }
        public static Bitmap Title(this Bitmap baseimage)
        {
            Bitmap finalcomposition = new Bitmap(baseimage.Width, baseimage.Height, PixelFormat.Format32bppArgb);
            using (Graphics g = Graphics.FromImage(finalcomposition))
            {
                // Use best quality rendering.
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
                // Draw in the base image.
                g.DrawImage(baseimage, new Point(0, 0));
                // Draw in the splash text.
                float text_top_size = 72;
                g.DrawString("RenderToy", new Font("Arial", text_top_size), Brushes.Black, 4, 4);
                float text_bottom_size = 16;
                SizeF text_bottom_rect = g.MeasureString("© 2016 Adam Johnson", new Font("Arial", text_bottom_size));
                g.DrawString("© 2016 Adam Johnson", new Font("Arial", text_bottom_size), Brushes.Black, baseimage.Width - text_bottom_rect.Width - 8, baseimage.Height - text_bottom_rect.Height - 8);
            }
            return finalcomposition;
        }
    }
}
