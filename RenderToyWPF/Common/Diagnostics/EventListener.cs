using System;
using System.Diagnostics.Tracing;

namespace RenderToy.Diagnostics
{
    public class RenderToyEventListener : EventListener
    {
        public RenderToyEventListener()
        {
            EnableEvents(RenderToyEventSource.Default, EventLevel.LogAlways);
        }
        protected override void OnEventWritten(EventWrittenEventArgs eventData)
        {
            try
            {
                HACKEventWritten(this, eventData);
            }
            catch
            {
            }
        }
        public event EventHandler<EventWrittenEventArgs> HACKEventWritten;
    }
}