////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Diagnostics;
using System;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;

namespace RenderToy.WPF
{
    class PerformanceView : FrameworkElement
    {
        DependencyProperty TimeBaseProperty = DependencyProperty.Register("TimeBase", typeof(double), typeof(PerformanceView), new FrameworkPropertyMetadata(0.0));
        public double TimeBase
        {
            get { return (double)GetValue(TimeBaseProperty); }
            set { SetValue(TimeBaseProperty, value); }
        }
        DependencyProperty TimeLengthProperty = DependencyProperty.Register("TimeLength", typeof(double), typeof(PerformanceView), new FrameworkPropertyMetadata(1.0 / 60.0));
        public double TimeLength
        {
            get { return (double)GetValue(TimeLengthProperty); }
            set { SetValue(TimeLengthProperty, value); }
        }
        public PerformanceView()
        {
            ClipToBounds = true;
            WeakEventManager<RenderToyEventListener, EventWrittenEventArgs>.AddHandler(listener, "HACKEventWritten", HandleEvent);
            var timer = new DispatcherTimer(TimeSpan.FromSeconds(5), DispatcherPriority.ApplicationIdle, (s, e) => { capturecurrent = capturelatest; InvalidateVisual(); }, Dispatcher);
            timer.Start();
        }
        void HandleEvent(object sender, EventWrittenEventArgs eventData)
        {
            var newcapture = recording.HandleEvent(sender, eventData);
            if (newcapture != null)
            {
                capturelatest = newcapture;
            }
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            var point = e.GetPosition(this);
            double scalestart = capturecurrent.tickstartframe;
            double scaleend = scalestart + Stopwatch.Frequency * TimeLength;
            var findtracks = capturecurrent.markertracks
                .Where((m, i) => point.Y >= i * 16 && point.Y <= i * 16 + 16);
            var findmarkers = findtracks
                .SelectMany(i => i)
                .Select(i => new { MinX = (i.TickStart - scalestart) * ActualWidth / (scaleend - scalestart), MaxX = (i.TickEnd - scalestart) * ActualWidth / (scaleend - scalestart), Name = i.Name });
            var matchmarkers = findmarkers
                .Where(i => point.X >= i.MinX && point.X <= i.MaxX);
            string name = matchmarkers.Select(i => i.Name).FirstOrDefault();
            if (name != null)
            {
                ToolTip = null;
                ToolTip = name;
            }
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawRectangle(Brushes.White, null, new Rect(0, 0, ActualWidth, ActualHeight));
            // Draw all markers.
            int y = 0;
            double scalestart = capturecurrent.tickstartframe;
            double scaleend = scalestart + Stopwatch.Frequency * TimeLength;
            foreach (var track in capturecurrent.markertracks)
            {
                foreach (var marker in track)
                {
                    double x1 = (marker.TickStart - scalestart) * ActualWidth / (scaleend - scalestart);
                    double w1 = (marker.TickEnd - marker.TickStart) * ActualWidth / (scaleend - scalestart);
                    drawingContext.DrawRectangle(brush_cpu_block, pen_cpu_block, new Rect(x1, y + 2, w1, 12));
                }
                y += 16;
            }
            // Draw 30Hz, 60Hz and 90Hz waterlines.
            {
                double timebase = TimeBase;
                double timelength = TimeLength;
                double x30 = (1.0 / 30.0 - TimeBase) * ActualWidth / TimeLength;
                drawingContext.DrawLine(pen_30Hz, new Point(x30, 0), new Point(x30, ActualHeight));
                DrawText(drawingContext, "30Hz", new Point(x30, 0), brush_30Hz);
                double x60 = (1.0 / 60.0 - TimeBase) * ActualWidth / TimeLength;
                drawingContext.DrawLine(pen_60Hz, new Point(x60, 0), new Point(x60, ActualHeight));
                DrawText(drawingContext, "60Hz", new Point(x60, 0), brush_60Hz);
                double x90 = (1.0 / 90.0 - TimeBase) * ActualWidth / TimeLength;
                drawingContext.DrawLine(pen_90Hz, new Point(x90, 0), new Point(x90, ActualHeight));
                DrawText(drawingContext, "90Hz", new Point(x90, 0), brush_90Hz);
            }
            // Draw FPS counter.
            {
                double milliseconds = (double)(capturecurrent.tickendframe - capturecurrent.tickstartframe) / Stopwatch.Frequency * 1000.0;
                var formattedtext = new FormattedText(milliseconds.ToString("0.0") + "ms (" + (1000.0 / milliseconds).ToString("0.0") + " FPS)", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 16, Brushes.Black);
                drawingContext.DrawText(formattedtext, new Point(0, 0));
            }
        }
        static void DrawText(DrawingContext drawingContext, string text, Point p, Brush brush)
        {
            var formattedtext = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, brush);
            drawingContext.DrawText(formattedtext, p);
        }
        static Brush brush_30Hz = Brushes.Red;
        static Pen pen_30Hz = new Pen(brush_30Hz, 1);
        static Brush brush_60Hz = Brushes.Orange;
        static Pen pen_60Hz = new Pen(brush_60Hz, 1);
        static Brush brush_90Hz = Brushes.Green;
        static Pen pen_90Hz = new Pen(brush_90Hz, 1);
        static Brush brush_cpu_block = Brushes.LightGray;
        static Pen pen_cpu_block = new Pen(Brushes.Black, 1);
        CaptureProcessed capturecurrent = new CaptureProcessed();
        CaptureProcessed capturelatest = new CaptureProcessed();
        RenderToyEventListener listener = new RenderToyEventListener();
        EventRecording recording = new EventRecording();
    }
}