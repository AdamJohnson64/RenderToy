////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;

namespace RenderToy
{
    public class Node
    {
        public Node(string name, ITransformed transform, IPrimitive primitive, Vector4D wirecolor, IMaterial material)
        {
            Name = name;
            Transform = transform;
            Primitive = primitive;
            WireColor = wirecolor;
            Material = material;
        }
        public readonly string Name;
        public readonly ITransformed Transform;
        public readonly IPrimitive Primitive;
        public readonly Vector4D WireColor;
        public readonly IMaterial Material;
        public IReadOnlyList<Node> Children
        {
            get { return children; }
        }
        List<Node> children = new List<Node>();
    }
    public class Scene
    {
        public static Scene Default
        {
            get
            {
                Scene scene = new Scene();
                scene.children.Add(new Node("Plane Ground", new TransformMatrix3D(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), Materials.LightGray, new CheckerboardMaterial(Materials.Black, Materials.White)));
                scene.children.Add(new Node("Sphere (Red)", new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-5, 1, 0)), new Sphere(), Materials.Red, Materials.PlasticRed));
                scene.children.Add(new Node("Sphere (Green)", new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-3, 1, 0)), new Sphere(), Materials.Green, Materials.PlasticGreen));
                scene.children.Add(new Node("Sphere (Blue)", new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-1, 1, 0)), new Sphere(), Materials.Blue, Materials.PlasticBlue));
                scene.children.Add(new Node("Sphere (Yellow)", new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+1, 1, 0)), new Sphere(), Materials.Yellow, Materials.PlasticYellow));
                scene.children.Add(new Node("Cube (Magenta)", new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+3, 1, 0)), new Cube(), Materials.Magenta, Materials.PlasticMagenta));
                scene.children.Add(new Node("Sphere (Cyan)", new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+5, 1, 0)), new Sphere(), Materials.Cyan, Materials.PlasticCyan));
                scene.children.Add(new Node("Sphere (Glass)", new TransformMatrix3D(MathHelp.CreateMatrixTranslate(0, 3, 0)), new Sphere(), Materials.Black, Materials.Glass));
                scene.children.Add(new Node("Triangle (Green)", new TransformMatrix3D(MathHelp.CreateMatrixTranslate(3, 2.1, 0)), new Triangle(), Materials.Green, Materials.PlasticGreen));
                return scene;
            }
        }
        public IReadOnlyList<Node> Children
        {
            get { return children; }
        }
        public void AddChild(Node node)
        {
            children.Add(node);
        }
        List<Node> children = new List<Node>();
    }
    public class TransformedObject
    {
        TransformedObject(Node node, Matrix3D transform)
        {
            Node = node;
            Transform = transform;
        }
        public readonly Node Node;
        public readonly Matrix3D Transform;
        public static IEnumerable<TransformedObject> Enumerate(Scene scene)
        {
            if (scene == null) yield break;
            foreach (Node root in scene.Children)
            {
                foreach (TransformedObject tobj in Enumerate(root, Matrix3D.Identity))
                {
                    yield return tobj;
                }
            }
        }
        static IEnumerable<TransformedObject> Enumerate(Node node, Matrix3D parenttransform)
        {
            Matrix3D localtransform = parenttransform * node.Transform.Transform;
            yield return new TransformedObject(node, localtransform);
            foreach (Node child in node.Children)
            {
                foreach (TransformedObject transformedchild in Enumerate(child, localtransform))
                {
                    yield return transformedchild;
                }
            }
        }
    }
}