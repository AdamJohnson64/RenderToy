////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System.Threading;
using System.Windows.Threading;

namespace RenderToy.Utility
{
    public class DispatcherHelper
    {
        // Do NOT call this function from a static constructor.
        // Static constructors block dispatchers which can cause a deadlock here.
        public static Dispatcher CreateDispatcher()
        {
            Dispatcher dispatcher = null;
            EventWaitHandle wait = new EventWaitHandle(false, EventResetMode.ManualReset);
            var thread = new Thread(() =>
            {
                SynchronizationContext.SetSynchronizationContext(new SynchronizationContext());
                dispatcher = Dispatcher.CurrentDispatcher;
                wait.Set();
                Dispatcher.Run();
            });
            thread.SetApartmentState(ApartmentState.MTA);
            thread.Start();
            wait.WaitOne();
            return dispatcher;
        }
    }
}
