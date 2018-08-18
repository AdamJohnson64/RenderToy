////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.DirectX;
using RenderToy.SceneGraph;
using System;
using System.Windows.Threading;

namespace RenderToy.OpenVR
{
    class Application
    {
        static int Main(string[] args)
        {
            try
            {
                Console.WriteLine("Initializing renderer...");
#if OPENVR_INSTALLED
                Direct3D11Helper.Initialize();
                OpenVRPump.CreateThread(TransformedObject.ConvertToSparseScene(TestScenes.DefaultScene));
                Dispatcher.Run();
#else
                Console.WriteLine("OpenVR is not available.");
#endif
                return 0;
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
                return -1;
            }
        }
    }
}
