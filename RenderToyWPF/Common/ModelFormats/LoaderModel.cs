using RenderToy.SceneGraph;
using System.Threading.Tasks;

namespace RenderToy.ModelFormat
{
    public static class LoaderModel
    {
        public static async Task<INode> LoadFromPathAsync(string path)
        {
            if (path.ToUpperInvariant().EndsWith(".BPT"))
            {
                return await LoaderBPT.LoadFromPathAsync(path);
            }
            else if (path.ToUpperInvariant().EndsWith(".OBJ"))
            {
                return await LoaderOBJ.LoadFromPathAsync(path);
            }
            else if (path.ToUpperInvariant().EndsWith(".PLY"))
            {
                return await LoaderPLY.LoadFromPathAsync(path);
            }
            else
            {
                return null;
            }
        }
    }
}