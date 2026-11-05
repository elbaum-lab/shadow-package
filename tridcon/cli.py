# Copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3
# See the LICENSE-GPL-3 for details

import sys
import os
import socket
import subprocess
import pathlib
import glob
import re
import importlib.metadata

import tempfile
import datetime
import argparse
import logging
import tqdm.auto as tqdm
import warnings
import mrcfile
import mrcfile.mrcinterpreter

import numpy as np

if __package__ is None or __package__ == '':
    from dec import set_num_threads, tomogram
    from _helpers import assrt, pxsize, non_negative_float, float_ge_one, get_ncpu
    from _mem import limit_memory, maxrss, log_memory
    from _thumbnail import save_png
else:
    from .dec import set_num_threads, tomogram, assrt
    from ._helpers import assrt, pxsize, non_negative_float, float_ge_one, get_ncpu
    from ._mem import limit_memory, maxrss, log_memory
    from ._thumbnail import save_png

myname='3dcon'
version=None

def basic_conf(argv):
    parser = argparse.ArgumentParser(description='Basic config parser', add_help=False)
    parser.add_argument('--config', type=str, required=False, default=myname+'.conf', help='Config file')
    basic,argv=parser.parse_known_args(argv)
    return(basic.config, argv)

def expose(p, iteration=None, wildcard=False):
    exp={}
    exp['negative']='N' if p.negative else 'n'
    exp['flip']=p.flip if p.flip else '-'
    exp['unit']='A' if p.unit=='angstrom' else 'P'
    exp['epsilon']=p.epsilon
    exp['lambda']=p.lamb
    exp['lenA']=p.lena
    exp['lenL']=p.lenl
    exp['lampos']=p.lampos
    exp['2D']=p.axis2d if p.axis2d else '-'
    exp['version']=version
    exp['psf']=os.path.basename(p.psf_name)
    exp['infile']=os.path.basename(p.infile_name)
    if wildcard:
        d='[0-9]'
        dd=d*2
        ddc=dd+':'
        exp['utc']=d*4+('-'+dd)*2+'T'+ddc*2+dd+'+'+ddc+dd       #  2025-03-26T12:34:58+00:00
        exp['iteration']=d*3
    else:
        exp['utc']=datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec='seconds')
        exp['iteration']='' if None is iteration else f'{iteration:03d}'
    return(exp)

def intermediate_save_spec(spec):
    def intermediate(iteration):
        if iteration in intermediate_iters:
            return True
        for n in intermediate_every:
            if iteration % n == 0:
                return True
        return False

    intermediate_iters = set()
    intermediate_every = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if item.startswith(':') or item.startswith('/'):
            try:
                n = int(item[1:])
                if n > 0:
                    intermediate_every.append(n)
                else:
                    raise argparse.ArgumentTypeError(f'{spec}: invalid interval "{item}", must be positive')
            except ValueError:
                raise argparse.ArgumentTypeError(f'{spec}: invalid interval "{item}", expected ":N" where N is a positive integer')
        else:
            try:
                iter_num = int(item)
                if iter_num > 0:
                    intermediate_iters.add(iter_num)
                else:
                    raise argparse.ArgumentTypeError(f'{spec}: invalid iteration number "{item}", must be positive')
            except ValueError:
                raise argparse.ArgumentTypeError(f'{spec}: invalid iteration number "{item}", expected a positive integer')
    return intermediate

def conf(argv=[], conf_file=myname+'.conf'):
    global version
    cargv=[]
    if conf_file:
        try:
            with open(conf_file) as f:
                for l in f:
                    l=l.strip()
                    if len(l)==0 or l[0]=='#':
                        continue
                    cargv.extend(re.split(r'=|\s+', '--'+l, maxsplit=1))
        except FileNotFoundError:
            pass
    cargv=cargv+argv

    parser = argparse.ArgumentParser(description=f'Deconvolution per Arigovindan et al [{version}]', usage=f'%(prog)s [--help] [--config {myname}.conf] [options]')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('rb'), default=sys.stdin.buffer, help='Source file name (stdin)')
    parser.add_argument('--psf', required=True, type=argparse.FileType('rb'), help='PSF file name')
    parser.add_argument('--output', type=str, required=True, help='Output MRC file name')
    parser.add_argument('--label', type=str, required=False, default=f'Processed by {myname} {version} at {{utc}}', help='MRC label')
    parser.add_argument('--thumbnail', metavar='THMBNL', type=str, required=False, default=None, help='Save thumbnail PNG of the central slice')
    parser.add_argument('--thmb-axis', type=str, choices=['x','y','z'], default='z', help='Axis for the thumbnail (default=z)')
    parser.add_argument('--reuse', action='store_true', help='Reuse existing output')
    parser.add_argument('--negative', action='store_true', help='Assume negated input/output')
    parser.add_argument('--illumination-factor', type=float_ge_one, default=1.0, help='With --negative, estimate incident illumination as this factor times the maximum input value (default=1)')
    parser.add_argument('--flip', type=str, default=None, choices=['x', 'y', 'z'], help='Flip PSF')
    parser.add_argument('--2D', metavar='AXIS2D', dest='axis2d', type=str, default=None, choices=['x', 'y', 'z'], help='Squeeze axis in PSF for 2D convolution')
    parser.add_argument('--psf-tola', metavar='ABS_TOL', type=non_negative_float, required=False, default=0, help='Tolerance for PSF autoshrink (absolute)')
    parser.add_argument('--psf-tolr', metavar='REL_TOL', type=non_negative_float, required=False, default=0, help='Tolerance for PSF autoshrink (relative)')

    parser.add_argument('--autosave', default=None, metavar='#SECONDS', dest='checkpoint_interval', type=int, help='Checkpoint interval')
    parser.add_argument('--intermediate', metavar="C,S,V", type=intermediate_save_spec, required=False, default='', help='Comma-separated list of iterations to save')

    parser.add_argument('--iterations', metavar='N', type=int, required=True, help='Max iterations')

    parser.add_argument('--epsilon', metavar="\u03b5", type=non_negative_float, required=False, default=1e-5, help='A small number to regularize log')
    parser.add_argument('--lambda', metavar="\u03bb", dest='lamb', type=non_negative_float, required=True, help='Regularization strength')

    parser.add_argument('--px-size', type=pxsize, required=False, help='Override pixel dimension from the input MRC (angstroms)')
    parser.add_argument('--unit', type=str, choices=['px','angstrom'], default='px', help='Units for the lengths below (px)')

    parser.add_argument('--lenA', metavar="A_LENGTH", dest='lena', type=non_negative_float, required=True, help='Arigovindan smoothing length')
    parser.add_argument('--lenL', metavar="L_LENGTH", dest='lenl', type=non_negative_float, required=True, help='Laplacian smoothing length')

    parser.add_argument('--lampos', metavar="\u03bb+", type=non_negative_float, required=True, help='Negativity penalty')
    parser.add_argument('--parallel', action='store_true', help='Run on available devices in parallel')
    ncpu = get_ncpu()
    parser.add_argument('--cpu', metavar='#CPU', type=int, required=False, default=ncpu, help=f'CPUs to use (default=all [{ncpu}])')
    parser.add_argument('--dev-reg', type=str, required=False, default='cpu', help='Device for regularization (default=cpu)')
    parser.add_argument('--dev-con', type=str, required=False, default='cpu', help='Device for convolutions (default=cpu)')
    parser.add_argument('--fft-mem', metavar='DEV_FFT', type=str, required=False, default='0', help='Optimize fft convolve for DEV_FFT memory')
    parser.add_argument('--memory', type=int, metavar='#CHUNKS' , required=False, const=9, default=None, action='store', nargs='?', help='Optimize for memory')
    parser.add_argument('--fft-thrsh', metavar='#SLICES', type=int, required=False, default=3, help='Threshold PSF thickness for fft convolve')    
    parser.add_argument('--mem-guard', metavar='GUARD', const=-1, default=False, nargs='?', type=int, help='Install OOM guardrail')
    parser.add_argument('--quiet', action='store_true', help='Disable progress bar')
    parser.add_argument('--verbose', action='store_true', help='Verbose progress bar')
    parser.add_argument('--dry-run', const=True, default=False, nargs='?', help='Check parameters and exit. Specify "psf" to test fitness')
    parser.add_argument('--debug', metavar='path', type=str, required=False, default=False, help=argparse.SUPPRESS)      # help='Record gradient and exit'

    args=parser.parse_args(cargv)
    if args.fft_mem.lower() in ('none','no','false','0'):
        args.fft_mem=False
    if args.thumbnail == '':
        args.thumbnail=None
    if args.checkpoint_interval is not None and args.checkpoint_interval <= 0:
        args.checkpoint_interval=None
    args.psf_name=args.psf.name
    args.infile_name=args.infile.name
    return(args)

def dxyz2dnp(header, dxyz):
    if header is None:
        axes = (1,2,3)
    else:
        axes = [header.mapc, header.mapr, header.maps]          # axes corresp to cols/rows/secs (1,2,3 for X,Y,Z)
    dnp = [dxyz[a-1] for a in axes[::-1]]
    return dnp

def dnp2dxyz(header, dnp):
    if header is None:
        axes = (1,2,3)
    else:
        axes = [header.mapc, header.mapr, header.maps]          # axes corresp to cols/rows/secs (1,2,3 for X,Y,Z)
    dxyz=[None,None,None]
    for a,l in zip(axes, dnp):
        dxyz[3-a]=l
    return dxyz


def xyz2axis(header, xyz):
    if header is None:
        axes={'x':2, 'y':1, 'z':0}
    else:
        axes = {'x':3-header.mapc, 'y':3-header.mapr, 'z':3-header.maps}
    return axes[xyz]

def main(argv=None):
    global version

    try:
        version='g-'+subprocess.check_output(["git", "describe", "--always", "--long"], cwd=pathlib.Path(__file__).resolve().parent, stderr=subprocess.DEVNULL).strip().decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            if __package__ is None or __package__ == '':
                from _version import __version__
            else:
                from ._version import __version__
            version=__version__
        except ModuleNotFoundError:
            try:
                version = 'm-'+importlib.metadata.version(myname)
            except (importlib.metadata.PackageNotFoundError,ValueError):
                version='v.UNK'

    def save(result, iteration, pbar):
        out_path=p.output.format(**expose(p, iteration))
        if p.verbose:
            pbar.set_description(desc=os.path.basename(out_path))
            pbar.refresh()
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(out_path)) as tf:
            with mrcfile.new(tf.name, overwrite=True) as output:
                if src_header is not None:
                    output.header[...]=src_header
                else:
                    output.header.label[0]='Source MRC header corrupted'
                output.header.cella.x, output.header.cella.y, output.header.cella.z = dnp2dxyz(output.header, (l*n for l,n in zip(lengthnp,result.shape)))
                if src_extended_header is not None:
                    output.set_extended_header(src_extended_header)
                if len(p.label)>0:
                    try:
                        l=p.label.format(**expose(p, iteration))
                        if len(l)>80:
                            pbar.warn('MRC label too long, truncating')
                            l=l[:80]
                        output.add_label(l)
                    except (ValueError, IndexError):
                        pbar.warn('Can not update MRC header')
                if p.negative:
                    result=-result
                output.set_data(result)
                if not output.is_volume():
                    pbar.warn('MRC is not 3D volume, fixing header')
                    output.set_volume()

        os.replace(tf.name, out_path)
        os.chmod(out_path, perms)

        if p.thumbnail:
            axis=xyz2axis(output.header, p.thmb_axis)
            slc = result.take(indices=result.shape[axis]//2, axis=axis)
            try:
                save_png(slc, p.thumbnail.format(**expose(p, iteration)))
            except ImportWarning as e:
                pbar.warn(f'Thumbnail not saved: {str(e)}')

        logging.info(f"{out_path} saved successfully  {maxrss()=}")
        return

    def excepthook(exc_type, exc_value, exc_traceback):
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        exc_name = exc_type.__name__
        exc_msg = str(exc_value) if exc_value else "No details available"
        print(f"{exc_name}: {exc_msg}", file=sys.stderr)
        return

    pbar_warnings_displayed = set()
    def pbar_warn(msg):
        logging.info(f'Progress bar warning: {msg}')
        if msg not in pbar_warnings_displayed:
            pbar_warnings_displayed.add(msg)
            pbar.write(f"Warning: {msg}")
        return

    logging.basicConfig(level=logging.INFO, filename=myname+'.log', format='%(asctime)s %(message)s')
    logging.captureWarnings(True)
    sys.excepthook=excepthook

    perms = os.umask(0o666)     # set umask to disable everything
    os.umask(perms)             # and restore it
    perms = 0o666 & ~perms

    logging.info(f"{myname} v:{version} started {sys.argv} at {socket.gethostname()} with {maxrss()=}")
    log_memory(logging.info)
    if None is argv:
        argv=sys.argv[1:]
    conf_file,argv =basic_conf(argv)

    p=conf(argv, conf_file=conf_file)
    logging.info(str(vars(p)))

    if p.mem_guard:
        limit_memory(logging.info, p.mem_guard)

    out_path=p.output.format(**expose(p, wildcard=True))
    logging.info('Output file is: '+out_path)

    set_num_threads(p.cpu)

    assrt(p.fft_thrsh>=0, 'fft-thrsh must be non-negative but got {p.fft_thrsh}', logging)

    psf=mrcfile.mrcinterpreter.MrcInterpreter(iostream=p.psf)

    assrt(not p.infile.isatty(), 'input MRC file is not read from console', logging)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            src=mrcfile.mrcinterpreter.MrcInterpreter(iostream=p.infile)
            src_extended_header=src.extended_header.copy()
            src_header=src.header.copy()
            s=src.header
            lengthxyz=[s.cella.x/s.nx, s.cella.y/s.ny, s.cella.z/s.nz]
            lengthnp=dxyz2dnp(s, lengthxyz)
        except ValueError as e:
            print(f"Error reading {p.infile.name}: {str(e)}", file=sys.stderr)
            assrt(p.px_size, 'Specify pixel geometry with --px-size', logging)
            p.infile.seek(0)
            src=mrcfile.mrcinterpreter.MrcInterpreter(iostream=p.infile, permissive=True)
            print(f"Reading in permissive mode", file=sys.stderr)
            src_extended_header=None
            src_header=None

    prefix=None
    if p.verbose:
        prefix=os.path.basename(p.infile_name)

    if p.px_size:
        lengthxyz=p.px_size
        lengthnp=dxyz2dnp(src_header, lengthxyz)

    logging.info(f'tomogram[{src.data.shape}] x psf[{psf.data.shape}]')    

    s=psf.header
    psf_lengthxyz=[s.cella.x/s.nx, s.cella.y/s.ny, s.cella.z/s.nz]
    if not np.allclose(lengthxyz, psf_lengthxyz, rtol=1e-3):
        msg=f"voxel size mismatch: {lengthxyz}!={[float(l) for l in psf_lengthxyz]}"
        logging.info(msg)
        if p.px_size:
            print(f"Warning: {msg}", file=sys.stderr)
        else:
            assrt(False, msg, logging)

    if p.unit=='px':
        length=None
    else:
        length=lengthnp

    p.alpha1=p.lena**4
    p.alpha2=p.lenl**4

    src=src.data.astype(dtype=np.float32, copy=True)
    if p.negative:
        # Estimate the unattenuated incident illumination from the brightest
        # measured voxel. A factor > 1 accounts for thick samples in which no
        # voxel reaches the true incident illumination.
        measured_max=float(src.max())
        illumination=p.illumination_factor*measured_max
        src=illumination-src
        scale=float(src.max())
        assrt(np.isfinite(scale) and scale>0,
              'Negative input has no valid intensity range after illumination correction', logging)
        src/=scale
    else:
        # Preserve the original positive-mode normalization.
        src-=src.min()
        scale=float(src.max())
        assrt(np.isfinite(scale) and scale>0,
              'Positive input has no valid intensity range', logging)
        src/=scale

    if p.flip is not None:
        psf=np.flip(psf.data, axis=xyz2axis(psf.header, p.flip)).copy()
    else:
        psf=psf.data.copy()

    assrt(psf.min()==0, "min(PSF) != 0")
    if p.axis2d is not None:
        p2d=xyz2axis(src_header, p.axis2d)
        psf=psf.sum(axis=p2d, keepdims=True)
        pad = [(0, 0)] * psf.ndim
        pad[p2d] = (1, 1)
        psf=np.pad(psf,pad)
    abs_tol=max(p.psf_tola, psf.max()*p.psf_tolr)
    psf[psf<abs_tol]=0
    psf/=psf.sum()

    p.infile.close()
    p.psf.close()

    try:
        out_path=max(glob.glob(out_path))    # raises ValueError if the list is empty
        output=mrcfile.open(out_path, mode='r')
    except (ValueError,FileNotFoundError):
        result=None
    else:
        with output:
            assrt(p.reuse, f"Output file exists, specify --reuse if intended", logging)
            if src_extended_header is not None:
                assrt(np.all(output.extended_header==src_extended_header), f"MRC extended header mismatch {out_path} vs {p.infile}", logging)
            assrt(output.data.shape == src.shape, f"MRC data shape mismatch {output.data.shape}@{out_path} vs {src.shape}", logging)
            s=output.header
            out_lengthxyz=[s.cella.x/s.nx, s.cella.y/s.ny, s.cella.z/s.nz]
            assrt(np.allclose(lengthxyz, out_lengthxyz, rtol=1e-3), f"voxel size mismatch: {lengthxyz}!={[float(l) for l in out_lengthxyz]}@{out_path}")

            logging.info(f'Reading in older result from {out_path}')
            if p.negative:
                result=-output.data.copy()
            else:
                result=output.data.copy()
            src_extended_header=output.extended_header.copy()
            src_header=output.header.copy()
            if p.verbose:
                prefix=os.path.basename(out_path)

    t=tomogram(src, psf, intermediate=p.intermediate, result=result, logging=logging.info, maxrss=maxrss,
                   lamb=p.lamb, lampos=p.lampos, alpha1=p.alpha1, alpha2=p.alpha2, eps=p.epsilon,
                   checkpoint=p.checkpoint_interval, split_chunks_J1=p.memory, fft_thrsh=p.fft_thrsh, fft_mem=p.fft_mem,
                   dev_con=p.dev_con, dev_reg=p.dev_reg, parallel=p.parallel,
                   length=length, dry_run=p.dry_run
                  )

    del src
    del psf
    del result

    if p.dry_run:
        print(t.dryrun(p.dry_run))
    elif p.debug:
        with mrcfile.new(p.debug, overwrite=True) as debug:
            loss,grad=t.debug()
            logging.info(f'debug {loss=}')
            print(f'debug {loss=}', file=sys.stderr)
            debug.set_data(grad)
    else:
        pbarfile=sys.stderr
        if not pbarfile.isatty() and sys.stdout.isatty():
            pbarfile=sys.stdout
        pbardisable=None        # tqdm docs: if set to None, disable on non-TTY.
        if p.quiet:
            pbardisable=True
        with tqdm.tqdm(total=p.iterations, disable=pbardisable, file=pbarfile, dynamic_ncols=True, desc=prefix) as pbar:
            pbar.warn = pbar_warn
            t.optimize(maxiter=p.iterations, save_callback=save, pbar=pbar, verbose=p.verbose)

    log_memory(logging.info)
    logging.info(f'{myname} finished with  {maxrss()=}')
    return
