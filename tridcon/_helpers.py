# Copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3
# See the LICENSE-GPL-3 for details

import sys
import os
from argparse import ArgumentTypeError
import platform

def get_system_info(**versions):
    """Return a list of strings with OS, Python interpreter, CPU, and any extra library versions."""
    lines = [
        f'OS: {platform.platform()}',
        f'Python: {sys.version}',
    ]
    # CPU model
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    lines.append('CPU: ' + line.split(':', 1)[1].strip())
                    break
    except OSError:
        proc = platform.processor()
        lines.append(f'CPU: {proc}' if proc else f'CPU: {platform.machine()} ({platform.architecture()[0]})')
    lines.extend(f'{k}: {v}' for k, v in versions.items())
    return lines

def float_ge_one(input):
    """Argparse type for floating-point values greater than or equal to one."""
    try:
        f = float(input)
    except (TypeError, ValueError):
        raise ArgumentTypeError(f'Value must be a number but got {input}')
    if f < 1:
        raise ArgumentTypeError(f'Value must be at least 1 but got {f}')
    return f

def get_ncpu():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()

def assrt(cond, msg, logging=None):
    if not cond:
        print(f"Error: {msg}", file=sys.stderr)
        if logging is not None:
            logging.error(msg)
        sys.exit(1)
    return

def pxsize(input, min=1e-5, max=1e8): # smaller than a bug ~1mm, bigger than a proton ~1fm
    ps = []
    for s in input.split(','):
        try:
            f = float(s)
        except:
            raise ArgumentTypeError(f'Pixel size must be a number but got {s}')
        if f < min or f > max:
            raise ArgumentTypeError(f'Pixel size must be in [{min},{max}] range but got {f}')
        ps.append(f)

    if len(ps)==1:
        ps *= 3
    elif len(ps) != 3:
        raise ArgumentTypeError(f'Exactly three pixel sizes are required but got {ps}')
    return(ps)

def non_negative_float(input):
    try:
        f = float(input)
    except:
        raise ArgumentTypeError(f'Value must be a number but got {input}')
    if f < 0:
        raise ArgumentTypeError(f'Value must be non-negative but got {f}')
    return f
