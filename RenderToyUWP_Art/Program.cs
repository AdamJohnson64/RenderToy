﻿using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

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
            CreateBitmap(48, 48, Path.Combine(cwd, "LockScreenLogo.scale-200.png"));
            CreateBitmap(1240, 600, Path.Combine(cwd, "SplashScreen.scale-200.png"));
            CreateBitmap(300, 300, Path.Combine(cwd, "Square150x150Logo.scale-200.png"));
            CreateBitmap(88, 88, Path.Combine(cwd, "Square44x44Logo.scale-200.png"));
            CreateBitmap(24, 24, Path.Combine(cwd, "Square44x44Logo.targetsize-24_altform-unplated.png"));
            CreateBitmap(50, 50, Path.Combine(cwd, "StoreLogo.png"));
            CreateBitmap(620, 300, Path.Combine(cwd, "Wide310x150Logo.scale-200.png"));
            return 0;
        }
        static void CreateBitmap(int width, int height, string filename)
        {
            Console.WriteLine("Creating bitmap at '" + filename + "' with dimensions (" + width + "," + height + ").");
            CreateBitmap(width, height).Save(filename);
        }
        static Bitmap CreateBitmap(int width, int height)
        {
            // Set up a simple scene with a red sphere.
            Scene scene = new Scene();
            scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixIdentity()), new Plane(), Materials.DarkGray, new CheckerboardMaterial(Materials.Black, Materials.White)));
            scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-2, 1, 0)), new Sphere(), Materials.Red, Materials.PlasticRed));
            scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(0, 1, 0)), new Sphere(), Materials.Green, Materials.PlasticGreen));
            scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+2, 1, 0)), new Sphere(), Materials.Blue, Materials.PlasticBlue));
            scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(0, 1, -2)), new Sphere(), Materials.Black, Materials.Glass));
            // Position our camera and build the inverse MVP.
            Matrix3D view = MathHelp.Invert(MathHelp.CreateMatrixLookAt(new Point3D(-1, 3, -4), new Point3D(0, 0, 0), new Vector3D(0, 1, 0)));
            Matrix3D proj = CameraPerspective.CreateProjection(0.001, 100.0, 45.0, 45.0);
            // Create the bitmap.
            Bitmap bitmap = new System.Drawing.Bitmap(width, height, PixelFormat.Format32bppArgb);
            BitmapData data = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            try
            {
                RenderToyCLI.RaytraceCPUF64AA(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(view * proj * CameraPerspective.AspectCorrectFit(width, height))), data.Scan0, width, height, data.Stride, 4, 4);
            }
            finally
            {
                bitmap.UnlockBits(data);
            }
            if (width > 1000 && height > 100)
            {
                using (Graphics g = Graphics.FromImage(bitmap))
                {
                    Brush brush_front = Brushes.White;
                    Brush brush_shadow = new SolidBrush(Color.FromArgb(64, 64, 64));
                    string text_tl = "RenderToy";
                    float text_tls = 72;
                    g.DrawString(text_tl, new Font("Arial", text_tls), brush_shadow, 8, 8);
                    g.DrawString(text_tl, new Font("Arial", text_tls), brush_front, 4, 4);
                    string text_br = "Because everyone should render a teapot...";
                    float text_brs = 24;
                    SizeF size = g.MeasureString(text_br, new Font("Arial", text_brs));
                    g.DrawString(text_br, new Font("Arial", text_brs), brush_shadow, width - size.Width - 4, height - size.Height - 4);
                    g.DrawString(text_br, new Font("Arial", text_brs), brush_front, width - size.Width - 8, height - size.Height - 8);
                }
            }
            return bitmap;
        }
    }
}
