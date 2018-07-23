////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Diagnostics.Tracing;

namespace RenderToy.Diagnostics
{
    [EventSource(Name = "RenderToy", Guid = "0bd0db6d-fb7b-4adf-9e04-866e249f26a4")]
    public class RenderToyEventSource : EventSource
    {
        RenderToyEventSource() : base("RenderToy", EventSourceSettings.ThrowOnEventWriteErrors)
        {
        }
        public const int _RenderBegin = 1;
        public const int _RenderEnd = 2;
        public const int _UpdateBegin = 3;
        public const int _UpdateEnd = 4;
        [Event(_RenderBegin, Message = "RenderBegin", Keywords = Keywords.Rendering, Level = EventLevel.LogAlways)]
        public void RenderBegin()
        {
            WriteEvent(_RenderBegin);
        }
        [Event(_RenderEnd, Message = "RenderEnd", Keywords = Keywords.Rendering, Level = EventLevel.LogAlways)]
        public void RenderEnd()
        {
            WriteEvent(_RenderEnd);
        }
        [Event(_UpdateBegin, Message = "RenderBegin", Keywords = Keywords.Rendering, Level = EventLevel.LogAlways)]
        public void UpdateBegin()
        {
            WriteEvent(_UpdateBegin);
        }
        [Event(_UpdateEnd, Message = "RenderEnd", Keywords = Keywords.Rendering, Level = EventLevel.LogAlways)]
        public void UpdateEnd()
        {
            WriteEvent(_UpdateEnd);
        }
        public class Keywords
        {
            public const EventKeywords Rendering = (EventKeywords)0x0001;
        }
        public static RenderToyEventSource Default = new RenderToyEventSource();
    }
    public class RenderToyEventListener : EventListener
    {
        public RenderToyEventListener()
        {
            EnableEvents(RenderToyEventSource.Default, EventLevel.LogAlways);
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
