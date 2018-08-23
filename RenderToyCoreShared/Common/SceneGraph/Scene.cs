////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.DirectX;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.Textures;
using RenderToy.Transforms;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows;

namespace RenderToy.SceneGraph
{
    public interface IScene
    {
        IReadOnlyList<INode> Children { get; }
    }
    public class Scene : IScene
    {
        public IReadOnlyList<INode> Children { get { return children; } }
        public readonly List<INode> children = new List<INode>();
    }
    public static class TestScenes
    {
        public static IScene DefaultScene1 { get; private set; }
        public static IScene DefaultScene2 { get; private set; }
        public static IScene DefaultScene3 { get; private set; }
        static TestScenes()
        {
            {
                Scene scene = new Scene();
                scene.children.Add(new Node("Plane Ground", new TransformMatrix(MathHelp.CreateMatrixScale(10, 10, 10)), Plane.Default, StockMaterials.LightGray, StockMaterials.Brick));
                scene.children.Add(new Node("Sphere (Red)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-5, 1, 0)), Sphere.Default, StockMaterials.Red, CreateColoredMaterial(StockMaterials.Red)));
                scene.children.Add(new Node("Sphere (Green)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-3, 1, 0)), Mesh.CreateMesh(Sphere.Default, 18, 9), StockMaterials.Green, CreateColoredMaterial(StockMaterials.Green)));
                scene.children.Add(new Node("Sphere (Blue)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-1, 1, 0)), Sphere.Default, StockMaterials.Blue, CreateColoredMaterial(StockMaterials.Blue)));
                scene.children.Add(new Node("Sphere (Yellow)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+1, 1, 0)), Sphere.Default, StockMaterials.Yellow, CreateColoredMaterial(StockMaterials.Yellow)));
                scene.children.Add(new Node("Cube (Magenta)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+3, 1, 0)), Cube.Default, StockMaterials.Magenta, StockMaterials.Brick));
                scene.children.Add(new Node("Sphere (Cyan)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+5, 1, 0)), Sphere.Default, StockMaterials.Cyan, CreateColoredMaterial(StockMaterials.Cyan)));
                scene.children.Add(new Node("Sphere (Glass)", new TransformMatrix(MathHelp.CreateMatrixTranslate(0, 3, 0)), Sphere.Default, StockMaterials.Black, StockMaterials.Glass));
                AddOpenVR(scene);
                DefaultScene1 = scene;
            }
            {
                Scene scene = new Scene();
                var materialBrick = new OBJMaterial();
                {
                    materialBrick.map_Kd = StockMaterials.Brick;
                    var displace = new MNSubtract
                    {
                        Lhs = new BrickMask { U = new MNMultiply { Lhs = new MNTexCoordU(), Rhs = new MNConstant { Value = 4 } }, V = new MNMultiply { Lhs = new MNTexCoordV(), Rhs = new MNConstant { Value = 4 } } },
                        Rhs = new MNMultiply { Lhs = new Perlin2D { U = new MNMultiply { Lhs = new MNTexCoordU(), Rhs = new MNConstant { Value = 512 } }, V = new MNMultiply { Lhs = new MNTexCoordV(), Rhs = new MNConstant { Value = 512 } } }, Rhs = new MNConstant { Value = 0.001 } }
                    };
                    materialBrick.map_bump = new BumpGenerate { U = new MNTexCoordU(), V = new MNTexCoordV(), Displacement = displace };
                    materialBrick.displacement = new MNVector4D { R = displace, G = displace, B = displace, A = new MNConstant { Value = 1 } };
                }
                scene.children.Add(new Node("Plane Ground", new TransformMatrix(MathHelp.CreateMatrixScale(10, 10, 10)), Plane.Default, StockMaterials.LightGray, materialBrick));
                scene.children.Add(new Node("Back Left", new TransformMatrix(MathHelp.CreateMatrixTranslate(-2, 1, -2)), Sphere.Default, StockMaterials.Red, materialBrick));
                scene.children.Add(new Node("Back Right", new TransformMatrix(MathHelp.CreateMatrixTranslate(2, 1, -2)), Sphere.Default, StockMaterials.Green, materialBrick));
                scene.children.Add(new Node("Front Left", new TransformMatrix(MathHelp.CreateMatrixTranslate(-2, 1, 2)), Sphere.Default, StockMaterials.Blue, materialBrick));
                scene.children.Add(new Node("Front Right", new TransformMatrix(MathHelp.CreateMatrixTranslate(2, 1, 2)), Sphere.Default, StockMaterials.White, materialBrick));
                AddOpenVR(scene);
                DefaultScene2 = scene;
            }
            {
                Scene scene = new Scene();
                var rootassembly = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                var roothdr = Path.Combine(rootassembly, "..\\..\\ThirdParty\\RenderToyAssets\\Environments\\uffizi_cross.hdr");
                var hdrimage = LoaderImage.LoadFromPath(roothdr);
                var hdrcube = new SurfaceCrossToCube(hdrimage);
                var transformControls =
                    MathHelp.CreateMatrixRotation(MathHelp.CreateQuaternionRotation(new Vector3D(1, 0, 0), -45)) *
                    MathHelp.CreateMatrixRotation(MathHelp.CreateQuaternionRotation(new Vector3D(0, 1, 0), -60)) *
                    MathHelp.CreateMatrixScale(0.05, 0.05, 0.05) *
                    MathHelp.CreateMatrixTranslate(-0.5, 1.25, 0.5);
                {
                    Matrix3D transform = MathHelp.CreateMatrixTranslate(0, 0, 1) * transformControls;
                    scene.children.Add(new Node("Cubemap Image", new TransformMatrix(transform), Plane.Default, StockMaterials.LightGray, hdrimage));
                }
                {
                    Matrix3D transform = new Matrix3D(
                        0.25, 0, 0, 0,
                        0, 0, -0.25, 0,
                        0, 0.25, 0, 0,
                        0, 1, 0.5, 1
                    );
                    scene.children.Add(new Node("Cubemap Image", new TransformMatrix(transform), Mesh.CreateMesh(Sphere.Default, 90, 90), StockMaterials.LightGray, hdrcube));
                }
                {
                    var facenames = new[] { "-X", "+X", "-Y", "+Y", "-Z", "+Z" };
                    for (int face = 0; face < 6; ++face)
                    {
                        Matrix3D transform = MathHelp.CreateMatrixTranslate((face - 2.5) * 2, 0, -1) * transformControls;
                        scene.children.Add(new Node("Cubemap Side " + facenames[face], new TransformMatrix(transform), Plane.Default, StockMaterials.LightGray, hdrcube.GetSurface(face, 0)));
                    }
                }
                AddOpenVR(scene);
                DefaultScene3 = scene;
            }
        }
        static IMaterial CreateColoredMaterial(Vector4D rgba)
        {
            return new OBJMaterial
            {
                map_Kd = new MNVector4D
                {
                    R = new MNConstant { Value = rgba.X },
                    G = new MNConstant { Value = rgba.Y },
                    B = new MNConstant { Value = rgba.Z },
                    A = new MNConstant { Value = rgba.W }
                }
            };
        }
        public static void AddOpenVR(Scene scene)
        {
#if OPENVR_INSTALLED
            try
            {
                ////////////////////////////////////////////////////////////////////////////////
                // Add in components for OpenVR.
                //var controllerModelLoaded = await LoaderModel.LoadFromPathAsync("C:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR\\resources\\rendermodels\\vr_controller_vive_1_5\\body.obj");
                //var controllerModel = controllerModelLoaded.Children[0];
                //IPrimitive controllerPrimitive = controllerModel.Primitive;
                //IMaterial controllerMaterial = controllerModel.Material;
                IPrimitive controllerPrimitive = null;
                IMaterial controllerMaterial = StockMaterials.PlasticWhite;
                scene.children.Add(new Node("OpenVR HMD", new TransformHMD(), controllerPrimitive, StockMaterials.White, controllerMaterial));
                scene.children.Add(new Node("OpenVR Left Controller", new TransformLeftHand(), controllerPrimitive, StockMaterials.White, controllerMaterial));
                scene.children.Add(new Node("OpenVR Right Controller", new TransformRightHand(), controllerPrimitive, StockMaterials.White, controllerMaterial));
                ////////////////////////////////////////////////////////////////////////////////
                // Add in the HMD camera.
                {
                    Matrix3D transform = new Matrix3D(
                        1, 0, 0, 0,
                        0, 0, -1, 0,
                        0, 1, 0, 0,
                        0, 1, 2, 1
                    );
                    //scene.children.Add(new Node("Left Eye Preview", new TransformMatrix(transform), Plane.Default, StockMaterials.LightGray, new MaterialOpenVRCameraDistorted()));
                }
                ////////////////////////////////////////////////////////////////////////////////
                // Add in a preview of the desktop on the right.
                {
                    var transformRightUI =
                        MathHelp.CreateMatrixScale(1920.0 / 1080.0, 1, 1) *
                        MathHelp.CreateMatrixScale(0.1, 0.1, 0.1) *
                        MathHelp.CreateMatrixRotation(MathHelp.CreateQuaternionRotation(new Vector3D(1, 0, 0), -45)) *
                        MathHelp.CreateMatrixRotation(MathHelp.CreateQuaternionRotation(new Vector3D(0, 1, 0), 60)) *
                        MathHelp.CreateMatrixTranslate(0.5, 1.25, 0.5);
                    scene.children.Add(new Node("Desktop", new TransformMatrix(transformRightUI), Plane.Default, StockMaterials.LightGray, new DXGIDesktopMaterial()));
                }
            }
            catch
            {
                Debug.WriteLine("WARNING: Unable to load controller model.");
            }
#endif
        }
    }
}