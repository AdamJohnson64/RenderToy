////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using System;

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
                var renderer = OpenVRPump.CreateRenderer(TransformedObject.ConvertToSparseScene(TestScenes.DefaultScene));
                Console.WriteLine("Render pump starting...");
                while (true)
                {
                    renderer();
                }
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
