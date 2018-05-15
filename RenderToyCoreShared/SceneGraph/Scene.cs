////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.Transforms;
using RenderToy.Utility;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.SceneGraph
{
    public class Scene
    {
        public static Scene Default
        {
            get
            {
                return scene;
            }
        }
        public IReadOnlyList<Node> Children
        {
            get { return children; }
        }
        static Scene()
        {
            scene = new Scene();
#if DEBUG
            var meshbvh = new Sphere();
#else
            var mesh = Mesh.CreateMesh(new Sphere(), 18, 9);
            var meshbvh = MeshBVH.Create(Mesh.FlattenIndices(mesh.Vertices, mesh.Triangles).ToArray());
#endif
            scene.children.Add(new Node("Plane Ground", new TransformMatrix(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), StockMaterials.LightGray, new MNCheckerboard()));
            scene.children.Add(new Node("Sphere (Red)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-5, 1, 0)), new Sphere(), StockMaterials.Red, StockMaterials.PlasticRed));
            scene.children.Add(new Node("Sphere (Green)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-3, 1, 0)), meshbvh, StockMaterials.Green, StockMaterials.PlasticGreen));
            scene.children.Add(new Node("Sphere (Blue)", new TransformMatrix(MathHelp.CreateMatrixTranslate(-1, 1, 0)), new Sphere(), StockMaterials.Blue, StockMaterials.PlasticBlue));
            scene.children.Add(new Node("Sphere (Yellow)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+1, 1, 0)), new Sphere(), StockMaterials.Yellow, StockMaterials.PlasticYellow));
            scene.children.Add(new Node("Cube (Magenta)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+3, 1, 0)), new Cube(), StockMaterials.Magenta, StockMaterials.PlasticMagenta));
            scene.children.Add(new Node("Sphere (Cyan)", new TransformMatrix(MathHelp.CreateMatrixTranslate(+5, 1, 0)), new Sphere(), StockMaterials.Cyan, StockMaterials.PlasticCyan));
            scene.children.Add(new Node("Sphere (Glass)", new TransformMatrix(MathHelp.CreateMatrixTranslate(0, 3, 0)), new Sphere(), StockMaterials.Black, StockMaterials.Glass));
        }
        public void AddChild(Node node)
        {
            children.Add(node);
        }
        List<Node> children = new List<Node>();
        static Scene scene;
    } 
}