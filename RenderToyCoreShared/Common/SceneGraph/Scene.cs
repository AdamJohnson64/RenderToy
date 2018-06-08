////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.Transforms;
using RenderToy.Utility;
using System.Collections.Generic;

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
            scene.children.Add(new Node("Plane Ground", new TransformMatrix(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), StockMaterials.LightGray, StockMaterials.MarbleTile));
            scene.children.Add(new Node("Sphere (Red)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-5, 1, 0)), new Sphere(), StockMaterials.Red, StockMaterials.PlasticRed));
            scene.children.Add(new Node("Sphere (Green)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-3, 1, 0)), Mesh.CreateMesh(new Sphere(), 18, 9), StockMaterials.Green, StockMaterials.PlasticGreen));
            scene.children.Add(new Node("Sphere (Blue)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-1, 1, 0)), new Sphere(), StockMaterials.Blue, StockMaterials.PlasticBlue));
            scene.children.Add(new Node("Sphere (Yellow)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+1, 1, 0)), new Sphere(), StockMaterials.Yellow, StockMaterials.PlasticYellow));
            scene.children.Add(new Node("Cube (Magenta)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+3, 1, 0)), new Cube(), StockMaterials.Magenta, StockMaterials.Brick));
            scene.children.Add(new Node("Sphere (Cyan)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+5, 1, 0)), new Sphere(), StockMaterials.Cyan, StockMaterials.PlasticCyan));
            scene.children.Add(new Node("Sphere (Glass)", new TransformMatrix(MathHelp.CreateMatrixTranslate(0, 3, 0)), new Sphere(), StockMaterials.Black, StockMaterials.Glass));
        }
        static Scene scene;
    }
}