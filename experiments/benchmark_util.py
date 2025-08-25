import sys, multiprocessing as mp, traceback, time

def measure_peak_rss_safe(func, *args, use_gpu=False, force_cpu=True, **kwargs):
    """
    CUDA-safe version of measure_peak_rss.
    
    Args:
        func: Function to measure
        force_cpu: If True, tries to move any torch tensors to CPU before subprocess
        use_gpu: Whether to measure GPU memory (only works in main process)
    """
    # Check if we have CUDA tensors that need special handling
    cuda_detected = False
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Simple heuristic to detect if we might have CUDA issues
            try:
                if torch.cuda.current_device() >= 0:
                    cuda_detected = True
            except:
                pass
    except ImportError:
        pass
    
    if cuda_detected and force_cpu:
        print("Warning: CUDA detected, running without subprocess memory measurement")
        # Run directly without subprocess to avoid CUDA context issues
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Try to get approximate memory info from main process
        peak_bytes = None
        try:
            import psutil, os
            peak_bytes = psutil.Process(os.getpid()).memory_info().rss
        except:
            pass
            
        gpu_peak = None
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_peak = torch.cuda.max_memory_reserved()
            except:
                pass
        
        return {
            "result": result, 
            "peak_rss_bytes": peak_bytes, 
            "peak_gpu_bytes": gpu_peak,
            "measured_in_subprocess": False
        }
    else:
        # Use original subprocess method
        result = measure_peak_rss(func, *args, use_gpu=use_gpu, **kwargs)
        result["measured_in_subprocess"] = True
        return result

def measure_peak_rss(func, *args, use_gpu=False, **kwargs):
    """
    Run `func(*args, **kwargs)` in a fresh subprocess and return:
      {
        'result': <func return>,
        'peak_rss_bytes': <peak resident set size of the subprocess>,
        'peak_gpu_bytes': <CUDA peak reserved bytes or None>
      }
    Notes:
      - Accurately captures native memory (NumPy, PyTorch, etc.).
      - On Linux, ru_maxrss is in KiB; on macOS it's in bytes. We normalize to bytes.
      - Function must be picklable (define it at module top level).
    """
    def _target(conn, func, args, kwargs, use_gpu):
        err = None
        result = None
        gpu_peak = None
        try:
            if use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            result = func(*args, **kwargs)

            if use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        # reserved is a better proxy for peak footprint
                        gpu_peak = torch.cuda.max_memory_reserved()
                except Exception:
                    gpu_peak = None

        except Exception:
            err = traceback.format_exc()

        # Peak RSS for this child process
        peak_bytes = None
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                peak_bytes = int(r)                        # bytes on macOS
            else:
                peak_bytes = int(r) * 1024                 # KiB -> bytes on Linux
        except Exception:
            # Fallback: approximate with current RSS if resource is unavailable
            try:
                import psutil, os
                peak_bytes = psutil.Process(os.getpid()).memory_info().rss
            except Exception:
                peak_bytes = None

        conn.send((result, err, peak_bytes, gpu_peak))
        conn.close()

    parent, child = mp.Pipe(duplex=False)
    p = mp.Process(target=_target, args=(child, func, args, kwargs, use_gpu))
    p.start()
    result, err, peak_bytes, gpu_peak = parent.recv()
    p.join()

    if err:
        raise RuntimeError(f"Subroutine raised an exception:\n{err}")

    return {"result": result, "peak_rss_bytes": peak_bytes, "peak_gpu_bytes": gpu_peak}


import os, time, threading
import psutil

class PeakRSS:
    """Track peak resident memory (bytes) during a with-block."""
    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak_rss = 0
        self.end_rss = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.proc = psutil.Process(os.getpid())
        self.peak_rss = self.proc.memory_info().rss
        def _poll():
            local_peak = self.peak_rss
            while not self._stop.is_set():
                rss = self.proc.memory_info().rss
                if rss > local_peak:
                    local_peak = rss
                time.sleep(self.interval)
            self.peak_rss = max(self.peak_rss, local_peak)
        self._t = threading.Thread(target=_poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._t.join()
        self.end_rss = self.proc.memory_info().rss