using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Threading;

namespace RenderToy.Diagnostics
{
    class EventRecording
    {
        public CaptureProcessed HandleEvent(object sender, EventWrittenEventArgs eventData)
        {
            recordingLock.WaitOne();
            try
            {
                if (eventData.EventId == RenderToyEventSource._RenderBegin)
                {
                    markersActive.Clear();
                    markersCollected.Clear();
                    stopwatch.Restart();
                    tickstartframe = stopwatch.ElapsedTicks;
                }
                if (eventData.EventId == RenderToyEventSource._RenderEnd)
                {
                    tickendframe = stopwatch.ElapsedTicks;
                    stopwatch.Stop();
                    CaptureProcessed processed = new CaptureProcessed();
                    processed.tickstartframe = tickstartframe;
                    processed.tickendframe = tickendframe;
                    // Bin markers into tracks and greedily fill them.
                    var markertracks = new List<List<MarkerRegion>>();
                    foreach (var marker in markersCollected)
                    {
                        //Find a track that can store this marker with no intersection.
                        var find = markertracks.FindAll(i => i[i.Count - 1].TickEnd <= marker.TickStart).ToArray();
                        if (find.Length == 0)
                        {
                            var newtrack = new List<MarkerRegion>();
                            newtrack.Add(marker);
                            markertracks.Add(newtrack);
                        }
                        else
                        {
                            find[0].Add(marker);
                        }
                    }
                    processed.markertracks = markertracks;
                    return processed;
                }
                if (eventData.EventId == RenderToyEventSource._MarkerBegin)
                {
                    var key = eventData.Payload[0];
                    var markerstart = stopwatch.ElapsedTicks;
                    markersActive.AddOrUpdate(key, (a) => markerstart, (a, b) => markerstart);
                }
                if (eventData.EventId == RenderToyEventSource._MarkerEnd)
                {
                    var markerend = stopwatch.ElapsedTicks;
                    var key = eventData.Payload[0];
                    long markerstart;
                    if (markersActive.TryRemove(key, out markerstart))
                    {
                        markersCollected.Add(new MarkerRegion { TickStart = markerstart, TickEnd = markerend, Name = key as string });
                    }
                }
                return null;
            }
            finally
            {
                recordingLock.ReleaseMutex();
            }
        }
        long tickstartframe;
        long tickendframe;
        List<MarkerRegion> markersCollected = new List<MarkerRegion>();
        Stopwatch stopwatch = new Stopwatch();
        ConcurrentDictionary<object, long> markersActive = new ConcurrentDictionary<object, long>();
        Mutex recordingLock = new Mutex();
    }
    struct MarkerRegion
    {
        public long TickStart;
        public long TickEnd;
        public string Name;
    }
    class CaptureProcessed
    {
        public long tickstartframe;
        public long tickendframe;
        public List<List<MarkerRegion>> markertracks = new List<List<MarkerRegion>>();
    }
}