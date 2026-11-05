# Copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3
# See the LICENSE-GPL-3 for details

import sys
import psutil

if sys.platform != 'win32':
    import resource

def fmt(n):
    if n>1e9:
        return f'{n/1e9:.3g}GB'
    else:
        return f'{n/1e6:.3g}MB'

def get_memory(log):
    available = []

    if sys.platform != 'win32':
        available = [m for m in resource.getrlimit(resource.RLIMIT_RSS) if m>0]

    # Get system memory info
    available.append(psutil.virtual_memory().available)

    # Try to get cgroup limits (Linux only)
    try:
        # Try cgroup v2 first
        try:
            with open('/sys/fs/cgroup/memory.current', 'r') as f:
                cgroup_current = int(f.read().strip())
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                limit_str = f.read().strip()
                if limit_str != 'max':
                    cgroup_limit = int(limit_str)
                    cgroup_available = cgroup_limit - cgroup_current
                    available.append(cgroup_available)
        except (FileNotFoundError, PermissionError):
            # Try cgroup v1
            with open('/proc/self/cgroup', 'r') as f:
                memory_cgroup_path = None
                for line in f:
                    if ':memory:' in line:
                        memory_cgroup_path = line.strip().split(':')[-1]
                        break
            
            if memory_cgroup_path:
                try:
                    usage_path = f'/sys/fs/cgroup/memory{memory_cgroup_path}/memory.usage_in_bytes'
                    limit_path = f'/sys/fs/cgroup/memory{memory_cgroup_path}/memory.limit_in_bytes'
                    
                    with open(usage_path, 'r') as f:
                        cgroup_current = int(f.read().strip())
                    with open(limit_path, 'r') as f:
                        cgroup_limit = int(f.read().strip())
                        # Very large values indicate no limit
                        if cgroup_limit < 2**60:
                            cgroup_available = cgroup_limit - cgroup_current
                            available.append(cgroup_available)
                except (FileNotFoundError, PermissionError):
                    pass
    
    except (FileNotFoundError, PermissionError, ValueError):
        # Not Linux or no cgroup info available
        pass

    avmem = min(available)
    available=[fmt(m) for m in available]
    log(f"min({available})={fmt(avmem)}")

    return avmem

def limit_memory(log, lim=-1):
    log('Installing OOM guardrail from '+maxrss())
    if lim<0:
        lim = get_memory(log)
    if sys.platform != 'win32':
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        log('Set AS limit: '+maxrss())
    else:
        log('Memory limit not supported on Windows')
    return

def log_memory(log):
    if sys.platform != 'win32':
        log(str(resource.getrusage(resource.RUSAGE_SELF)))
    else:
        log(str(psutil.Process().memory_info()))

def maxrss():
    if sys.platform != 'win32':
        rss=[m for m in resource.getrlimit(resource.RLIMIT_RSS) if m>0]
        if len(rss)==0:
            rss='?'
        else:
            rss=fmt(min(rss))

        ass=[m for m in resource.getrlimit(resource.RLIMIT_AS) if m>0]
        if len(ass)==0:
            ass=''
        else:
            ass='~'+fmt(min(ass))

        # ru_maxrss is in kilobytes on Linux, bytes on macOS
        maxrss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin':
            maxrss_bytes *= 1024
        return(fmt(maxrss_bytes)+'/'+rss+ass)
    else:
        mi = psutil.Process().memory_info()
        # peak_wset is the Windows equivalent of ru_maxrss (peak working set)
        return(fmt(mi.peak_wset)+'/?')




