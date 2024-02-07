import tracemalloc
import os

# list to store memory snapshots
snaps = []
 
def snapshot():
    snaps.append(tracemalloc.take_snapshot())
 
 
def display_stats():
    with open( os.path.expanduser("~/alloc_log.txt"), "a") as outfile:
        stats = snaps[0].statistics('filename')
        print("\n*** top 5 stats grouped by filename ***")
        outfile.write("\n*** top 5 stats grouped by filename ***")
        for s in stats[:5]:
            print(s)
            outfile.write(f"\n{str(s)}")
    
 
 
def compare():
    with open( os.path.expanduser("~/alloc_log.txt"), "a") as outfile:
        first = snaps[0]
        for snapshot in snaps[1:]:
            stats = snapshot.compare_to(first, 'lineno')
            print("\n*** top 10 stats ***")
            outfile.write("\n*** top 10 stats ***")
            for s in stats[:10]:
                print(s)
                outfile.write(f"\n{str(s)}")
 
 
def print_trace():
    with open( os.path.expanduser("~/alloc_log.txt"), "a") as outfile:
        # pick the last saved snapshot, filter noise
        snapshot = snaps[-1].filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
        tracemalloc.Filter(False, "<unknown>"),
        ))
        largest = snapshot.statistics("traceback")[0]
     
        print(f"\n*** Trace for largest memory block - ({largest.count} blocks, {largest.size/1024} Kb) ***")
        outfile.write(f"\n*** Trace for largest memory block - ({largest.count} blocks, {largest.size/1024} Kb) ***")
        for l in largest.traceback.format():
            print(l)
            outfile.write(f"\n{str(l)}")

def reset():
    open(os.path.expanduser("~/alloc_log.txt"), "w").close()
