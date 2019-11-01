using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Transforms;
using System.Collections;
using System.Collections.Generic;

namespace RenderToy.DocumentModel
{
    public class SparseScene : IReadOnlyList<TransformedObject>
    {
        public TransformedObject this[int index] => new TransformedObject(
            TableNodeMaterial[IndexToNodeMaterial[index]],
            TableNodePrimitive[IndexToNodePrimitive[index]],
            TableNodeTransform[IndexToNodeTransform[index]],
            TableNodeWireColor[IndexToNodeMaterial[index]],
            TableTransform[IndexToTransform[index]]
        );
        public int Count => IndexToTransform.Count;
        public IEnumerator<TransformedObject> GetEnumerator()
        {
            int count = IndexToTransform.Count;
            for (int i = 0; i < count; ++i)
            {
                yield return this[i];
            }
        }
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable<TransformedObject>)this).GetEnumerator();
        }
        public readonly List<Mesh> TableMesh = new List<Mesh>();
        public readonly List<int> IndexToMesh = new List<int>();
        public readonly List<Matrix3D> TableTransform = new List<Matrix3D>();
        public readonly List<int> IndexToTransform = new List<int>();
        public readonly List<IMaterial> TableNodeMaterial = new List<IMaterial>();
        public readonly List<int> IndexToNodeMaterial = new List<int>();
        public readonly List<IPrimitive> TableNodePrimitive = new List<IPrimitive>();
        public readonly List<int> IndexToNodePrimitive = new List<int>();
        public readonly List<ITransform> TableNodeTransform = new List<ITransform>();
        public readonly List<int> IndexToNodeTransform = new List<int>();
        public readonly List<Vector4D> TableNodeWireColor = new List<Vector4D>();
        public readonly List<int> IndexToNodeWireColor = new List<int>();
    }
}