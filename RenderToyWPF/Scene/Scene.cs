////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public class Node
    {
        public Node(ITransformed transform, object primitive, Color wirecolor, IMaterial material)
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
        public Color WireColor
        {
            get { return wirecolor; }
        }
        public IReadOnlyList<Node> Children
        {
            get { return children; }
        }
        public readonly ITransformed transform;
        public readonly object primitive;
        public readonly Color wirecolor;
        public readonly IMaterial material;
        List<Node> children = new List<Node>();
    }
    public class Scene
    {
        public static Scene Default
        {
            get
            {
                Scene scene = new Scene();
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), Colors.LightGray, new CheckerboardMaterial(Colors.Black, Colors.White)));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(-2, 1, 0)), new Sphere(), Colors.Red, new ConstantColorMaterial(Colors.Red)));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(0, 1, 0)), new Sphere(), Colors.Green, new ConstantColorMaterial(Colors.Green)));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(+2, 1, 0)), new Sphere(), Colors.Blue, new ConstantColorMaterial(Colors.Blue)));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateMatrixTranslate(0, 3, 0)), new Sphere(), Colors.Black, new GlassMaterial()));
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
    class TransformedObject
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