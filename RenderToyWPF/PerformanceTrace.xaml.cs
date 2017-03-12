////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using RenderToy.Linq;

namespace RenderToy
{
    public class PerformanceTrackControl : FrameworkElement
    {
        #region - Section : Dependency Properties -
        public static DependencyProperty TraceTextProperty = DependencyProperty.Register("TraceText", typeof(string), typeof(PerformanceTrackControl));
        public string TraceText
        {
            get { return (string)GetValue(TraceTextProperty); }
            set { SetValue(TraceTextProperty, value); }
        }
        public static DependencyProperty TraceEventsProperty = DependencyProperty.Register("TraceEvents", typeof(IReadOnlyList<PerformanceEvent>), typeof(PerformanceTrackControl));
        public IReadOnlyList<PerformanceEvent> TraceEvents
        {
            get { return (IReadOnlyList<PerformanceEvent>)GetValue(TraceEventsProperty); }
            set { SetValue(TraceEventsProperty, value); }
        }
        public static DependencyProperty TraceBinsProperty = DependencyProperty.Register("TraceBins", typeof(IReadOnlyList<PerformanceTrack>), typeof(PerformanceTrackControl), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));
        public IReadOnlyList<PerformanceTrack> TraceBins
        {
            get { return (IReadOnlyList<PerformanceTrack>)GetValue(TraceBinsProperty); }
            set { SetValue(TraceBinsProperty, value); }
        }
        public static DependencyProperty TraceTimeMinProperty = DependencyProperty.Register("TraceTimeMin", typeof(long), typeof(PerformanceTrackControl), new FrameworkPropertyMetadata((long)0, FrameworkPropertyMetadataOptions.AffectsRender));
        public long TraceTimeMin
        {
            get { return (long)GetValue(TraceTimeMinProperty); }
            set { SetValue(TraceTimeMinProperty, value); }
        }
        public static DependencyProperty TraceTimeMaxProperty = DependencyProperty.Register("TraceTimeMax", typeof(long), typeof(PerformanceTrackControl), new FrameworkPropertyMetadata((long)0, FrameworkPropertyMetadataOptions.AffectsRender));
        public long TraceTimeMax
        {
            get { return (long)GetValue(TraceTimeMaxProperty); }
            set { SetValue(TraceTimeMaxProperty, value); }
        }
        #endregion
        #region - Section : Construction -
        public PerformanceTrackControl()
        {
            // Normally we pass a trace text from a textbox to fill the control.
            // This converter will take the input and parse it into an enumerable set of events.
            SetBinding(TraceEventsProperty, new Binding
            {
                RelativeSource = new RelativeSource(RelativeSourceMode.Self),
                Path = new PropertyPath(TraceTextProperty),
                Converter = new Converter((text) => PerformanceHelp.ReadEvents((string)text).ToList())
            });
            // This control cannot directly render events, it prefers binned tracks.
            // This converter will take the events and convert them into binned block ranges.
            SetBinding(TraceBinsProperty, new Binding
            {
                RelativeSource = new RelativeSource(RelativeSourceMode.Self),
                Path = new PropertyPath(TraceEventsProperty),
                Converter = new Converter((events) => PerformanceHelp.GetBins((IEnumerable<PerformanceEvent>)events))
            });
            // Determine the minimum time from the provided trace bins.
            SetBinding(TraceTimeMinProperty, new Binding
            {
                RelativeSource = new RelativeSource(RelativeSourceMode.Self),
                Path = new PropertyPath(TraceBinsProperty),
                Converter = new Converter((bins) => ((IEnumerable<PerformanceTrack>)bins).SelectMany(track => track.Events).Select(ev => ev.Begin.Timestamp).MinOrDefault())
            });
            // Determine the maximum time from the provided trace bins.
            SetBinding(TraceTimeMaxProperty, new Binding
            {
                RelativeSource = new RelativeSource(RelativeSourceMode.Self),
                Path = new PropertyPath(TraceBinsProperty),
                Converter = new Converter((bins) => ((IEnumerable<PerformanceTrack>)bins).SelectMany(track => track.Events).Select(ev => ev.End.Timestamp).MaxOrDefault())
            });
        }
        #endregion
        #region - Section : FrameworkElement Overrides -
        protected override void OnMouseRightButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseRightButtonDown(e);
            if (dragscrollenable) return;
            CaptureMouse();
            dragscrollenable = true;
            dragscrolltime = (long)(TraceTimeMin + (TraceTimeMax - TraceTimeMin) * (e.GetPosition(this).X / ActualWidth));
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!dragscrollenable) return;
            // Scroll the point at the mouse cursor to the time in "dragscrolltime".
            var tracemin = TraceTimeMin;
            var tracemax = TraceTimeMax;
            var tracelen = tracemax - tracemin;
            var pointer_coord = e.GetPosition(this);
            var pointer_offsety = pointer_coord.X / ActualWidth;
            tracemin = (long)(dragscrolltime - tracelen * pointer_offsety);
            tracemax = tracemin + tracelen;
            TraceTimeMin = tracemin;
            TraceTimeMax = tracemax;
        }
        protected override void OnMouseRightButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseUp(e);
            if (!dragscrollenable) return;
            ReleaseMouseCapture();
            dragscrollenable = false;
            dragscrolltime = 0;
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            // Record the mouse position and intended time at the mouse position.
            var tracemin = TraceTimeMin;
            var tracemax = TraceTimeMax;
            var tracelen = tracemax - tracemin;
            var pointer_coord = e.GetPosition(this);
            var pointer_offsety = pointer_coord.X / ActualWidth;
            var pointer_time = tracemin + tracelen * pointer_offsety;
            // Zooming In.
            if (e.Delta > 0)
            {
                // First zoom without preserving the mouse offset.
                tracemax = tracemin + tracelen / 2;
                tracelen = tracemax - tracemin;
                // Then scroll such that the time under the mouse is the same as when we started.
                tracemin = (long)(pointer_time - tracelen * pointer_offsety);
                tracemax = tracemin + tracelen;
                TraceTimeMin = tracemin;
                TraceTimeMax = tracemax;
            }
            if (e.Delta < 0)
            {
                // First zoom without preserving the mouse offset.
                tracemax = tracemin + tracelen * 2;
                tracelen = tracemax - tracemin;
                // Then scroll such that the time under the mouse is the same as when we started.
                tracemin = (long)(pointer_time - tracelen * pointer_offsety);
                tracemax = tracemin + tracelen;
                TraceTimeMin = tracemin;
                TraceTimeMax = tracemax;
            }
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            // Draw a transparent background so hit-tests will always land in this control.
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            var tracks = TraceBins;
            if (tracks == null) return;
            long timemin = TraceTimeMin;
            long timemax = TraceTimeMax;
            // Render the performance data using the most attrociously bone-headed slow method known to humankind.
            int trackcount = 0;
            Pen pen_black = new Pen(Brushes.Black, -1);
            foreach (var track in tracks)
            {
                foreach (var block in track.Events)
                {
                    double x1 = (block.Begin.Timestamp - timemin) * ActualWidth / (timemax - timemin);
                    double x2 = (block.End.Timestamp - timemin) * ActualWidth / (timemax - timemin);
                    if (x2 - x1 < 1) continue;
                    if (x2 < 0 || x1 > ActualWidth) continue;
                    double y1 = (trackcount + 0) * ActualHeight / tracks.Count;
                    double y2 = (trackcount + 1) * ActualHeight / tracks.Count;
                    var rect = new Rect(x1, y1, x2 - x1, y2 - y1);
                    drawingContext.DrawRectangle(Brushes.LightGray, pen_black, rect);
                    if (x2 - x1 < 32) continue;
                    drawingContext.PushClip(new RectangleGeometry(new Rect(x1, y1, x2 - x1, y2 - y1)));
                    drawingContext.DrawText(new FormattedText(block.Begin.Text, System.Globalization.CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 10, Brushes.Black), new Point(x1, y1));
                    drawingContext.Pop();
                }
                ++trackcount;
            }
        }
        bool dragscrollenable = false;
        long dragscrolltime = 0;
        #endregion
    }
    public partial class PerformanceTrace : UserControl
    {
        public PerformanceTrace()
        {
            InitializeComponent();
        }
    }
    public class Converter : IValueConverter
    {
        public Converter(Func<object, object> convert)
        {
            Convert = convert;
        }
        object IValueConverter.Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return Convert(value);
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
        Func<object, object> Convert;
    }
}

namespace RenderToy.Linq
{
    public static class Extensions
    {
        public static long MaxOrDefault(this IEnumerable<long> data)
        {
            long? found = null;
            foreach (var t in data)
            {
                found = Math.Max(found ?? t, t);
            }
            return found ?? 0;
        }
        public static long MinOrDefault(this IEnumerable<long> data)
        {
            long? found = null;
            foreach (var t in data)
            {
                found = Math.Min(found ?? t, t);
            }
            return found ?? 0;
        }
    }
}