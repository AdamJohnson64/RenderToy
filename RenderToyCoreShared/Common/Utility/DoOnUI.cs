////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Windows.Threading;

namespace RenderToy.Utility
{
    public class DoOnUI
    {
        public static void Call(Action func) { Dispatcher.Invoke(func); }
        public static T Call<T>(Func<T> func) { return Dispatcher.Invoke(func); }
        public static readonly Dispatcher Dispatcher = Dispatcher.CurrentDispatcher;
    }
}