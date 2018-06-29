////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Diagnostics.Tracing;
using System.Windows;

namespace RenderToy.WPF
{
    [EventSource(Name = "RenderToy", Guid = "0bd0db6d-fb7b-4adf-9e04-866e249f26a4")]
    public class RenderToyETWEventSource : EventSource
    {
        RenderToyETWEventSource() : base("RenderToy", EventSourceSettings.ThrowOnEventWriteErrors)
        {
        }
        public const int RenderBegin = 1;
        public const int RenderEnd = 2;
        [Event(RenderBegin, Message = "RenderBegin", Keywords = Keywords.Rendering, Level = EventLevel.LogAlways)]
        public void BeginFrame()
        {
            WriteEvent(RenderBegin);
        }
        [Event(RenderEnd, Message = "RenderEnd", Keywords = Keywords.Rendering, Level = EventLevel.LogAlways)]
        public void EndFrame()
        {
            WriteEvent(RenderEnd);
        }
        public class Keywords
        {
            public const EventKeywords Rendering = (EventKeywords)0x0001;
        }
        public static RenderToyETWEventSource Default = new RenderToyETWEventSource();
    }
    public class RenderToyETWListener : EventListener
    {
        public RenderToyETWListener()
        {
            EnableEvents(RenderToyETWEventSource.Default, EventLevel.LogAlways);
        }
        protected override void OnEventWritten(EventWrittenEventArgs eventData)
        {
            if (HACKEventWritten != null)
            {
                HACKEventWritten(this, eventData);
            }
        }
        public event EventHandler<EventWrittenEventArgs> HACKEventWritten;
    }
}
