using System.Collections.Generic;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public class Node
    {
        public Node(ITransformed transform, object primitive, Color wirecolor)
        {
            this.transform = transform;
            this.primitive = primitive;
            this.wirecolor = wirecolor;
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
        ITransformed transform;
        object primitive;
        Color wirecolor;
        List<Node> children = new List<Node>();
    }
    public class Scene
    {
        public static Scene Default
        {
            get
            {
                Scene scene = new Scene();
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateScaleMatrix(10, 10, 10)), new Plane(), Colors.LightGray));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateTranslateMatrix(-2, 1, 0)), new Sphere(), Colors.Red));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateTranslateMatrix(0, 1, 0)), new Sphere(), Colors.Green));
                scene.children.Add(new Node(new TransformMatrix3D(MathHelp.CreateTranslateMatrix(+2, 1, 0)), new Sphere(), Colors.Blue));
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