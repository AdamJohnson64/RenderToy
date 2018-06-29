////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
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
            WeakEventManager<RenderToyETWListener, EventWrittenEventArgs>.AddHandler(listener, "HACKEventWritten", HandleEvent);
        }
        void HandleEvent(object sender, EventWrittenEventArgs eventData)
        {
            if (eventData.EventId == RenderToyETWEventSource.RenderBegin)
            {
                datetimestart = DateTime.Now;
            }
            if (eventData.EventId == RenderToyETWEventSource.RenderEnd)
            {
                datetimeend = DateTime.Now;
                InvalidateVisual();
            }
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            TimeSpan timespan = datetimeend.Subtract(datetimestart);
            var formattedtext = new FormattedText(timespan.TotalMilliseconds.ToString("0.0") + "ms (" + (1000 / timespan.TotalMilliseconds).ToString("0.0") + " FPS)", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Green);
            drawingContext.DrawText(formattedtext, new Point(0, 0));
        }
        RenderToyETWListener listener = new RenderToyETWListener();
        DateTime datetimestart;
        DateTime datetimeend;
    }
}