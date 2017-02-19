////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;

namespace RenderToy
{
    public class Node
    {
        public Node(ITransformed transform, object primitive, Point4D wirecolor, IMaterial material)
        {
            this.transform = transform;
            this.primitive = primitive;
            this.wirecolor = wirecolor;
            this.material = material;
        }
        public ITransformed Transform
        {
            get { return transform; }
        }
        public object Primitive
        {
            get { return primitive; }
        }
        public Point4D WireColor
        {
            get { return wirecolor; }
        }
        public IReadOnlyList<Node> Children
        {
            get { return children; }
        }
        public readonly ITransformed transform;
        public readonly object primitive;
        public readonly Point4D wirecolor;
        public readonly IMaterial material;
        List<Node> children = new List<Node>();
    }
    public class Scene
    {
        static Mesh bunny = MeshPLY.LoadFromPath("bun_zipper_res4.ply");
        public static Scene Default
        {
            get
            {
                Scene scene = new Scene();
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), Materials.LightGray, new CheckerboardMaterial(Materials.Black, Materials.White)));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-5, 1, 0)), new Sphere(), Materials.Red, Materials.PlasticRed));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-3, 1, 0)), new Sphere(), Materials.Green, Materials.PlasticGreen));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-1, 1, 0)), new Sphere(), Materials.Blue, Materials.PlasticBlue));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+1, 1, 0)), new Sphere(), Materials.Yellow, Materials.PlasticYellow));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+3, 1, 0)), new Cube(), Materials.Magenta, Materials.PlasticMagenta));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+5, 1, 0)), new Sphere(), Materials.Cyan, Materials.PlasticCyan));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(0, 3, 0)), new Sphere(), Materials.Black, Materials.Glass));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(3, 2.1, 0)), new Triangle(), Materials.Green, Materials.PlasticGreen));
                if (bunny != null)
                {
                    scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixScale(100, 100, 100) * MathHelp.CreateMatrixTranslate(0, 0, 2)), bunny, Materials.LightGray, Materials.PlasticRed));
                }
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
        public Matrix3D Transform;
        public Node Node;
        public static IEnumerable<TransformedObject> Enumerate(Scene scene)
        {
            foreach (Node root in scene.Children)
            {
                foreach (TransformedObject tobj in Enumerate(root, Matrix3D.Identity))
                {
                    yield return tobj;
                }
            }
        }
        private static IEnumerable<TransformedObject> Enumerate(Node node, Matrix3D parenttransform)
        {
            Matrix3D localtransform = parenttransform * node.Transform.Transform;
            yield return new TransformedObject { Transform = localtransform, Node = node };
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