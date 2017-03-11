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
using System.Windows.Media;

namespace RenderToy
{
    public class PerformanceTrackControl : FrameworkElement
    {
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
        public PerformanceTrackControl()
        {
            SetBinding(TraceEventsProperty, new Binding
            {
                RelativeSource = new RelativeSource(RelativeSourceMode.Self),
                Path = new PropertyPath(TraceTextProperty),
                Converter = new Converter((text) => PerformanceHelp.ReadEvents((string)text).ToList())
            });
            SetBinding(TraceBinsProperty, new Binding
            {
                RelativeSource = new RelativeSource(RelativeSourceMode.Self),
                Path = new PropertyPath(TraceEventsProperty),
                Converter = new Converter((events) => PerformanceHelp.GetBins((IEnumerable<PerformanceEvent>)events))
            });
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            var events = TraceEvents;
            if (events == null) return;
            if (events.Count < 2) return;
            long timemin = events[0].Timestamp;
            long timemax = events[events.Count - 1].Timestamp;
            var tracks = TraceBins;
            if (tracks == null) return;
            int trackcount = 0;
            // Render the performance data using the most attrociously bone-headed slow method known to humankind.
            Pen pen_black = new Pen(Brushes.Black, -1);
            foreach (var track in tracks)
            {
                foreach (var block in track.Events)
                {
                    double x1 = (block.Begin.Timestamp - timemin) * ActualWidth / (timemax - timemin);
                    double x2 = (block.End.Timestamp - timemin) * ActualWidth / (timemax - timemin);
                    if (x2 - x1 < 64) continue;
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
