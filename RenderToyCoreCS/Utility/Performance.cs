////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace RenderToy
{
    public class Performance
    {
        public static void LogEvent(string text)
        {
#if !WINDOWS_UWP
            Console.WriteLine("@" + Stopwatch.GetTimestamp() + " THREADID " + Thread.CurrentThread.ManagedThreadId + " EVENT \"" + text + "\"");
#endif
        }
        public static void LogBegin(string text)
        {
#if !WINDOWS_UWP
            Console.WriteLine("@" + Stopwatch.GetTimestamp() + " THREADID " + Thread.CurrentThread.ManagedThreadId + " BEGIN \"" + text + "\"");
#endif
        }
        public static void LogEnd(string text)
        {
#if !WINDOWS_UWP
            Console.WriteLine("@" + Stopwatch.GetTimestamp() + " THREADID " + Thread.CurrentThread.ManagedThreadId + " END \"" + text + "\"");
#endif
        }
    }
}