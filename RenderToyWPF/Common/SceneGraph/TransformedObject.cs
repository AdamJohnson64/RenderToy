﻿using RenderToy.DocumentModel;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Primitives;
using RenderToy.Transforms;
using System.Collections.Generic;

namespace RenderToy.SceneGraph
{
    public class TransformedObject
    {
        public TransformedObject(IMaterial nodematerial, IPrimitive nodeprimitive, ITransform nodetransform, Vector4D nodewirecolor, Matrix3D transform)
        {
            TransformParent = Matrix3D.Identity;
            Transform = transform;
            NodeMaterial = nodematerial;
            NodePrimitive = nodeprimitive;
            NodeTransform = nodetransform;
            NodeWireColor = nodewirecolor;
        }
        public readonly Matrix3D TransformParent;
        public readonly Matrix3D Transform;
        public readonly IMaterial NodeMaterial;
        public readonly IPrimitive NodePrimitive;
        public readonly ITransform NodeTransform;
        public readonly Vector4D NodeWireColor;
        public static IEnumerable<TransformedObject> Enumerate(IScene scene)
        {
            if (scene == null) yield break;
            foreach (INode root in scene.Children)
            {
                foreach (TransformedObject tobj in Enumerate(root, Matrix3D.Identity))
                {
                    yield return tobj;
                }
            }
        }
        static IEnumerable<TransformedObject> Enumerate(INode node, Matrix3D parenttransform)
        {
            Matrix3D localtransform = parenttransform * node.Transform.Transform;
            yield return new TransformedObject(node.Material, node.Primitive, node.Transform, node.WireColor, localtransform);
            foreach (INode child in node.Children)
            {
                foreach (TransformedObject transformedchild in Enumerate(child, localtransform))
                {
                    yield return transformedchild;
                }
            }
        }
        public static SparseScene ConvertToSparseScene(IScene scene)
        {
            var output = new SparseScene();
            foreach (var t in Enumerate(scene))
            {
                output.IndexToNodeMaterial.Add(output.TableNodeMaterial.Count);
                output.TableNodeMaterial.Add(t.NodeMaterial);
                output.IndexToNodePrimitive.Add(output.TableNodePrimitive.Count);
                output.TableNodePrimitive.Add(t.NodePrimitive);
                output.IndexToNodeTransform.Add(output.TableNodeTransform.Count);
                output.TableNodeTransform.Add(t.NodeTransform);
                output.IndexToNodeWireColor.Add(output.TableNodeWireColor.Count);
                output.TableNodeWireColor.Add(t.NodeWireColor);
                output.IndexToTransform.Add(output.TableTransform.Count);
                output.TableTransform.Add(t.Transform);
            }
            return output;
        }
    }
}