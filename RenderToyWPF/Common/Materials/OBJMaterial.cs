using RenderToy.Textures;
using RenderToy.Utility;

namespace RenderToy.Materials
{
    public class OBJMaterial : ITexture, INamed
    {
        public string Name
        {
            get { return _name; }
            set { _name = value; }
        }
        public IMaterial map_Ka
        {
            get { return _map_Ka; }
            set { _map_Ka = value; }
        }
        public IMaterial map_Kd
        {
            get { return _map_Kd; }
            set { _map_Kd = value; }
        }
        public IMaterial map_d
        {
            get { return _map_d; }
            set { _map_d = value; }
        }
        public IMaterial map_bump
        {
            get { return _map_bump; }
            set { _map_bump = value; }
        }
        public IMaterial bump
        {
            get { return _bump; }
            set { _bump = value; }
        }
        public IMaterial displacement
        {
            get { return _displacement; }
            set { _displacement = value; }
        }
        public string GetName()
        {
            return _name;
        }
        public bool IsConstant()
        {
            return
                (_map_Ka == null ? true : _map_Ka.IsConstant()) &&
                (_map_Kd == null ? true : _map_Kd.IsConstant()) &&
                (_map_d == null ? true : _map_d.IsConstant()) &&
                (_map_bump == null ? true : _map_bump.IsConstant()) &&
                (_bump == null ? true : _bump.IsConstant());
        }
        public int GetTextureArrayCount()
        {
            return 1;
        }
        public int GetTextureLevelCount()
        {
            var kd = _map_Kd as ITexture;
            return kd == null ? 0 : kd.GetTextureLevelCount();
        }
        public ISurface GetSurface(int array, int level)
        {
            var kd = _map_Kd as ITexture;
            return kd == null ? null : kd.GetSurface(array, level);
        }
        string _name;
        IMaterial _map_Ka;
        IMaterial _map_Kd;
        IMaterial _map_d;
        IMaterial _map_bump;
        IMaterial _bump;
        IMaterial _displacement;
    }
}