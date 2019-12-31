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
        public const int _MarkerBegin = 3;
        public const int _MarkerEnd = 4;
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
        [Event(_MarkerBegin, Message = "MarkerBegin", Keywords = Keywords.Rendering, Level = EventLevel.LogAlways)]
        public void MarkerBegin(string name)
        {
            WriteEvent(_MarkerBegin, name);
        }
        [Event(_MarkerEnd, Message = "MarkerEnd", Keywords = Keywords.Rendering, Level = EventLevel.LogAlways)]
        public void MarkerEnd(string name)
        {
            WriteEvent(_MarkerEnd, name);
        }
        public class Keywords
        {
            public const EventKeywords Rendering = (EventKeywords)0x0001;
        }
        public static RenderToyEventSource Default = new RenderToyEventSource();
    }
}
