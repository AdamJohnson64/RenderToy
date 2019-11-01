using RenderToy.Materials;
using RenderToyCOM;
using System;
using System.Threading;

namespace RenderToy.DirectX
{
    public static class DXGIHelper
    {
        public static void Initialize()
        {
            IntPtr dxgiduplication = new IntPtr();
            Direct3D11Helper.Dispatcher.Invoke(() =>
            {
                var dxgifactory = DXGI.CreateDXGIFactory2();
                IDXGIAdapter1 dxgiadapter;
                dxgifactory.EnumAdapters1(0, out dxgiadapter);
                DXGI_ADAPTER_DESC1 dxgiadapterdesc;
                dxgiadapter.GetDesc1(out dxgiadapterdesc);
                IDXGIOutput dxgioutput = null;
                dxgiadapter.EnumOutputs(0, ref dxgioutput);
                IDXGIOutput6 dxgioutput6 = (IDXGIOutput6)dxgioutput;
                dxgiduplication = DXGI.IDXGIOutput1_DuplicateOutput(dxgioutput6, Direct3D11Helper.d3d11Device);
                D3D11_TEXTURE2D_DESC d3d11texture_desktopcopy = new D3D11_TEXTURE2D_DESC();
                d3d11texture_desktopcopy.Width = (uint)1920;
                d3d11texture_desktopcopy.Height = (uint)1080;
                d3d11texture_desktopcopy.MipLevels = 1;
                d3d11texture_desktopcopy.ArraySize = 1;
                d3d11texture_desktopcopy.Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM;
                d3d11texture_desktopcopy.SampleDesc.Count = 1;
                d3d11texture_desktopcopy.Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT;
                d3d11texture_desktopcopy.BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_SHADER_RESOURCE;
                unsafe
                {
                    D3D11_SUBRESOURCE_DATA* d3d11initialdata = null;
                    Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11texture_desktopcopy, ref *d3d11initialdata, ref d3d11texture_copieddesktop);
                }
                var vdesc = new D3D11_SHADER_RESOURCE_VIEW_DESC();
                vdesc.Format = DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
                vdesc.ViewDimension = D3D_SRV_DIMENSION.D3D11_SRV_DIMENSION_TEXTURE2D;
                vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.TextureCube.MipLevels = 1;
                vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.TextureCube.MostDetailedMip = 0;
                Direct3D11Helper.d3d11Device.CreateShaderResourceView(d3d11texture_copieddesktop, vdesc, ref d3d11srv_copieddesktop);
            });
            var thread = new Thread(() =>
            {
                while (true)
                {
                    Direct3D11Helper.Dispatcher.Invoke(() =>
                    {
                        DXGI_OUTDUPL_FRAME_INFO dxgiframeinfo = new DXGI_OUTDUPL_FRAME_INFO();
                        var dxgiresource = DXGI.IDXGIOutputDuplication_AcquireNextFrame(dxgiduplication, 0, ref dxgiframeinfo);
                        if (dxgiresource != null)
                        {
                            if (dxgiframeinfo.LastMouseUpdateTime.QuadPart != 0)
                            {
                                mouse = dxgiframeinfo.PointerPosition.Position;
                            }
                            var d3d11texture_currentdesktop = (ID3D11Texture2D)dxgiresource;
                            ID3D11DeviceContext d3d11devicecontext = null;
                            Direct3D11Helper.d3d11Device.GetImmediateContext(ref d3d11devicecontext);
                            d3d11devicecontext.CopyResource(d3d11texture_copieddesktop, d3d11texture_currentdesktop);
                            DXGI.IDXGIOutputDuplication_ReleaseFrame(dxgiduplication);
                        }
                    });
                    Thread.Sleep(1000 / 60);
                }
            });
            thread.SetApartmentState(ApartmentState.MTA);
            thread.Start();
        }
        static ID3D11Texture2D d3d11texture_copieddesktop = null;
        public static ID3D11ShaderResourceView d3d11srv_copieddesktop = null;
        public static tagPOINT mouse;
    }
    class DXGIDesktopMaterial : IMaterial
    {
        public bool IsConstant()
        {
            return false;
        }
    }
}
