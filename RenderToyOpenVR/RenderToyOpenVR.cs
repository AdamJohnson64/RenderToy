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
                var renderer = OpenVRPump.CreateRenderer(TransformedObject.Enumerate(TestScenes.DefaultScene));
                Console.WriteLine("Render pump starting...");
                while (true)
                {
                    renderer();
                }
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
