////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Diagnostics;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Globalization;
using System.Windows;
using System.Windows.Media;

namespace RenderToy.WPF
{
    class PerformanceView : FrameworkElement
    {
        public PerformanceView()
        {
            WeakEventManager<RenderToyEventListener, EventWrittenEventArgs>.AddHandler(listener, "HACKEventWritten", HandleEvent);
        }
        void HandleEvent(object sender, EventWrittenEventArgs eventData)
        {
            if (eventData.EventId == RenderToyEventSource._RenderBegin)
            {
                tickmarkerstart.Clear();
                markers.Clear();
                stopwatch.Restart();
                tickstartframe = stopwatch.ElapsedTicks;
            }
            if (eventData.EventId == RenderToyEventSource._RenderEnd)
            {
                tickendframe = stopwatch.ElapsedTicks;
                stopwatch.Stop();
                Dispatcher.Invoke(InvalidateVisual);
            }
            if (eventData.EventId == RenderToyEventSource._MarkerBegin)
            {
                var key = eventData.Payload[0];
                var markerstart = stopwatch.ElapsedTicks;
                tickmarkerstart.AddOrUpdate(key, (a) => markerstart, (a, b) => markerstart);
            }
            if (eventData.EventId == RenderToyEventSource._MarkerEnd)
            {
                var markerend = stopwatch.ElapsedTicks;
                var key = eventData.Payload[0];
                long markerstart;
                if (tickmarkerstart.TryGetValue(key, out markerstart))
                {
                    markers.Add(new MarkerRegion { TickStart = markerstart, TickEnd = markerend, Name = key as string });
                }
            }
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var copy = markers.ToArray();
            foreach (var marker in copy)
            {
                double x1 = (marker.TickStart - tickstartframe) * ActualWidth / (tickendframe - tickstartframe);
                double w1 = (marker.TickEnd - marker.TickStart) * ActualWidth / (tickendframe - tickstartframe);
                drawingContext.DrawRectangle(Brushes.Blue, null, new Rect(x1, 0, w1, ActualHeight));
            }
            {
                double milliseconds = (double)(tickendframe - tickstartframe) / Stopwatch.Frequency * 1000.0;
                var formattedtext = new FormattedText(milliseconds.ToString("0.0") + "ms (" + (1000.0 / milliseconds).ToString("0.0") + " FPS)", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 16, Brushes.Black);
                drawingContext.DrawText(formattedtext, new Point(0, 0));
            }
        }
        RenderToyEventListener listener = new RenderToyEventListener();
        Stopwatch stopwatch = new Stopwatch();
        long tickstartframe;
        long tickendframe;
        ConcurrentDictionary<object, long> tickmarkerstart = new ConcurrentDictionary<object, long>();
        struct MarkerRegion
        {
            public long TickStart;
            public long TickEnd;
            public string Name;
        }
        List<MarkerRegion> markers = new List<MarkerRegion>();
    }
}