////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;

namespace RenderToy.Utility
{
    public class DoOnUI
    {
        public static void Call(Action func) { wait.WaitOne(); Dispatcher.Invoke(func); }
        public static T Call<T>(Func<T> func) { wait.WaitOne(); return Dispatcher.Invoke(func); }
        public static Dispatcher Dispatcher = null;
        static EventWaitHandle wait = new EventWaitHandle(false, EventResetMode.ManualReset);
        static DoOnUI()
        {
            if (SynchronizationContext.Current == null)
            {
                SynchronizationContext.SetSynchronizationContext(new SynchronizationContext());
            }
            var thread = new Thread(() =>
            {
                Dispatcher = Dispatcher.CurrentDispatcher;
                wait.Set();
                Dispatcher.Run();
            });
            thread.SetApartmentState(ApartmentState.MTA);
            thread.Start();
        }
    }
}