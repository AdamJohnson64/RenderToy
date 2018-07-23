////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Diagnostics;
using System;
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
                stopwatch.Restart();
            }
            if (eventData.EventId == RenderToyEventSource._RenderEnd)
            {
                stopwatch.Stop();
                InvalidateVisual();
            }
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            double milliseconds = (double)stopwatch.ElapsedTicks / Stopwatch.Frequency * 1000.0;
            var formattedtext = new FormattedText(milliseconds.ToString("0.0") + "ms (" + (1000.0 / milliseconds).ToString("0.0") + " FPS)", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Green);
            drawingContext.DrawText(formattedtext, new Point(0, 0));
        }
        RenderToyEventListener listener = new RenderToyEventListener();
        Stopwatch stopwatch = new Stopwatch();
        DateTime datetimestart;
        DateTime datetimeend;
    }
}