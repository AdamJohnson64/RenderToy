////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;
using System.Threading;
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
                // Force the dispatcher onto this thread (we are a surrogate UI thread).
                var dispatcher = DoOnUI.Dispatcher;
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
