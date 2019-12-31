// The WorkQueue is a helper object which permits the execution of mini tasks (delegate Action).
// Tasks may spawn other tasks as they progress and the worker will terminate when all work is complete.
//
// We use this design as a means for investigating large-scale threading models in RenderToy.

using RenderToy.Diagnostics;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace RenderToy.Utility
{
    public class WorkQueue
    {
        public void Queue(Action work)
        {
            // Count the new work and queue for execution.
            Interlocked.Increment(ref workqueued);
            worklist.Enqueue(work);
        }
        public void Start()
        {
            // Create a set of workers.
            var workers = new List<Task>();
            Performance.LogEvent("Starting work on " + Environment.ProcessorCount + " logical CPUs");
            for (int i = 0; i < Environment.ProcessorCount; ++i)
            {
                int j = i;
                var task = Task.Run(async () => {
                    Performance.LogBegin("WorkQueue Executor " + j);
                    Performance.LogEvent("Remaining work: " + workqueued + " queued, " + workretire + " retired");
                TRYAGAIN:
                    // Try to dequeue some work or spin for more work.
                    Action work;
                    if (!worklist.TryDequeue(out work))
                    {
                        // If there's no work to do and no work running then there's nothing left to do; exit.
                        if (workexeced == 0) goto DONE;
                        // Take a short break and go around.
                        await Task.Delay(10);
                        goto TRYAGAIN;
                    }
                    // Increment the active worker count and process the work.
                    Interlocked.Increment(ref workexeced);
                    work();
                    Interlocked.Decrement(ref workexeced);
                    // Retire this work item.
                    Interlocked.Increment(ref workretire);
                    goto TRYAGAIN;
                DONE:
                    Performance.LogEnd("WorkQueue Executor " + j);
                });
                workers.Add(task);
            }
            // Wait for all workers to finish.
            Task.WaitAll(workers.ToArray());
        }
        /// <summary>
        /// Queue of pending work to be executed.
        /// </summary>
        ConcurrentQueue<Action> worklist = new ConcurrentQueue<Action>();
        /// <summary>
        /// Number of work items currently processing (in-flight).
        /// </summary>
        int workexeced = 0;
        /// <summary>
        /// Total number of work items queued so far.
        /// </summary>
        int workqueued = 0;
        /// <summary>
        /// Total number of work items executed and retired so far.
        /// </summary>
        int workretire = 0;
    }
}