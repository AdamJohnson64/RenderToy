﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.Transforms;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

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
        public static IScene DefaultScene
        {
            get { return scene; }
        }
        static TestScenes()
        {
            scene = new Scene();
            var materialBrick = new LoaderOBJ.OBJMaterial();
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
            scene.children.Add(new Node("Sphere (Red)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-5, 1, 0)), Sphere.Default, StockMaterials.Red, StockMaterials.PlasticRed));
            scene.children.Add(new Node("Sphere (Green)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-3, 1, 0)), Mesh.CreateMesh(Sphere.Default, 18, 9), StockMaterials.Green, StockMaterials.PlasticGreen));
            scene.children.Add(new Node("Sphere (Blue)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-1, 1, 0)), Sphere.Default, StockMaterials.Blue, StockMaterials.PlasticBlue));
            scene.children.Add(new Node("Sphere (Yellow)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+1, 1, 0)), Sphere.Default, StockMaterials.Yellow, StockMaterials.PlasticYellow));
            scene.children.Add(new Node("Cube (Magenta)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+3, 1, 0)), Cube.Default, StockMaterials.Magenta, StockMaterials.Brick));
            scene.children.Add(new Node("Sphere (Cyan)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+5, 1, 0)), Sphere.Default, StockMaterials.Cyan, StockMaterials.PlasticCyan));
            scene.children.Add(new Node("Sphere (Glass)", new TransformMatrix(MathHelp.CreateMatrixTranslate(0, 3, 0)), Sphere.Default, StockMaterials.Black, StockMaterials.Glass));
            try
            {
                var openvr = new OpenVRHelper();
                var controller = LoaderOBJ.LoadFromPath("C:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR\\resources\\rendermodels\\vr_controller_vive_1_5\\body.obj").First();
                scene.children.Add(new Node("OpenVR HMD", new TransformHMD(openvr), controller.Primitive, StockMaterials.White, controller.Material));
                scene.children.Add(new Node("OpenVR Left Controller", new TransformLeftHand(openvr), controller.Primitive, StockMaterials.White, controller.Material));
                scene.children.Add(new Node("OpenVR Right Controller", new TransformRightHand(openvr), controller.Primitive, StockMaterials.White, controller.Material));
                Matrix3D transform = new Matrix3D(
                    1, 0, 0, 0,
                    0, 0, -1, 0,
                    0, 1, 0, 0,
                    0, 1, 2, 1
                );
                //scene.children.Add(new Node("Left Eye Preview", new TransformMatrix(transform), Plane.Default, StockMaterials.LightGray, new MaterialOpenVRCameraDistorted(openvr)));
            }
            catch
            {
                Debug.WriteLine("WARNING: Unable to load controller model.");
            }
        }
        static Scene scene;
    }
}