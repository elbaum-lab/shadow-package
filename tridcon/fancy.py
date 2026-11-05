#!/usr/bin/env python
#
# Copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3
# See the LICENSE-GPL-3 for details

import sys
import numpy as np
import math
import argparse
import mrcfile
import tqdm.auto as tqdm

if __package__ is None or __package__ == '':
    from airy import Grid as AiryGrid, Beam as AiryBeam
    from _helpers import pxsize, get_ncpu
else:
    from .airy import Grid as AiryGrid, Beam as AiryBeam
    from ._helpers import pxsize, get_ncpu

eps=1e-8
class Grid:
    def __init__(self, shape, pxsize, dtype=np.float32, tilt_axis=1):
        self.dtype=dtype
        self.shape=np.flip(np.asarray(shape))
        self.pxsize=np.flip(np.asarray(pxsize, dtype=dtype))
        self.tilt_axis=tilt_axis
        self.airy=AiryGrid(self.shape, self.pxsize, dtype=dtype, ncpu=get_ncpu())

    def fanout(self, beam, tilts=None, weighted=False):
        if not tilts:
            tilts=[0]

        for theta, phi in (pbar:=tqdm.tqdm(tilts, desc='Tilts')):
                if phi is None:
                    n=np.asarray([0, math.sin(theta), math.cos(theta)])
                    n=np.roll(n, self.tilt_axis)
                else: # [z, y, x]
                    n=np.asarray([math.cos(theta),
                                  math.sin(theta)*math.sin(phi),
                                  math.sin(theta)*math.cos(phi)])
                if weighted:
                    w=math.cos(theta)
                else:
                    w=1.0
                beam.beam(self.airy, direction=n, weight=w, write=pbar.write)

        return(self.airy.array)

def autoshrink(a):
    for i in range(a.ndim):
        b=np.swapaxes(a,i,0)
        while b.shape[0]>4 and not np.any(b[[0,1,-2,-1]]):
            b=b[1:-1]
        a=np.swapaxes(b,i,0)
    return(a)


def main():
    parser = argparse.ArgumentParser(description='Generate a 3D fanout', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', type=str, required=True, help='Output file name')

    parser.add_argument('--ref', metavar='TOM', type=argparse.FileType('rb'), default=None, help='Use the geometry of the tomogram as reference')
    parser.add_argument('--shape', metavar='N', type=int, nargs=3, help='Shape of the grid in voxels')
    parser.add_argument('--px-size', metavar='P', type=pxsize, required=False, help='Voxel dimensions in angstroms')

    parser.add_argument('--tilt-axis', type=str, default='y', choices=['x', 'y', 'z'], help='Tilt axis')
    parser.add_argument('--neg-angle', action='store_true', help='Negate tilt angles')
    parser.add_argument('--base-angle', metavar='θ₀', type=float, default=0, help='Tilt angles offset')
    parser.add_argument('--tilts', type=argparse.FileType('r'), default='-', help='Path to a text file with one tilt (degrees) per line')

    parser.add_argument('--wavelength', metavar='λ', type=float, required=True, help='Beam wavelength')
    parser.add_argument('--alpha', metavar='α', type=float, required=True, help='Beam semi-divergence angle')

    parser.add_argument('--rabs', metavar='R', type=float, default=None, help='Absolute radius for the PSF mask')
    parser.add_argument('--rrel', metavar='r', type=float, default=10., help='Relative radius for the PSF mask in units of the beam waist')
    parser.add_argument('--density', metavar='d', type=float, default=4., help='Subpixel density for the beam rendering')

    parser.add_argument('--weighted', action='store_true', default=False, help='Weight the beam intensity by the cosine of the tilt angle')
    parser.add_argument('--crop', action='store_true', default=False, help='Crop the PSF to non-zero region before saving')
    args = parser.parse_args()

    neg_angle=-1. if args.neg_angle else 1.
    base_angle=args.base_angle
    tilt_axes = {'x':2, 'y':1, 'z':0}
    tilt_axis=tilt_axes[args.tilt_axis]

    tilts = []
    flag_polar = False
    for line in args.tilts:
        if line.lstrip().startswith('#'):
            continue
        line.replace(',', ' ')
        t=[(neg_angle * math.pi / 180) * float(x) for x in line.split()]
        if len(t)==0:
            continue
        elif len(t)==1:
            tilts.append([ t[0] + base_angle * math.pi / 180 , None ])
        elif len(t)==2:
            flag_polar = True
            tilts.append(t)
        else:
            raise ValueError('Each line in tilts file must have one or two values')

    if args.tilts is not sys.stdin:
        args.tilts.close()

    if args.ref is None and (args.shape is None or args.px_size is None):
        parser.error('either --ref or both --shape and --px-size must be provided')

    w0=args.wavelength/args.alpha

    if args.rabs is not None:
        rmax=args.rabs
    else:
        rmax=args.rrel*w0

    if args.ref is not None:
        src=mrcfile.mrcinterpreter.MrcInterpreter(iostream=args.ref, header_only=True)
        h=src.header
        if args.px_size is None:
            args.px_size=[h.cella.x/h.nx, h.cella.y/h.ny, h.cella.z/h.nz]

        if args.shape is None:
            args.shape=[1+2*int(h.nx), 1+2*int(h.ny), 1+2*int(h.nz)]
            if not flag_polar:
                args.shape[2-tilt_axis]=math.ceil(rmax/args.px_size[2-tilt_axis])*2+3  # ensure the kernel is not tight against the edge
        args.ref.close()

    label=f'{len(tilts)} {"weighted " if args.weighted else ""}{'' if flag_polar else args.tilt_axis+'-'}tilts for l={args.wavelength}, a={args.alpha}, d={args.density}, rmax={rmax:.3g}'
    print(f'Generating {args.shape} grid with {args.px_size[0]:.3g}x{args.px_size[1]:.3g}x{args.px_size[2]:.3g}Å pixel size using {label}... ',
           file=sys.stderr, flush=True, end='')
    grid = Grid(shape=args.shape, pxsize=args.px_size, tilt_axis=tilt_axis)
    print(' OK', file=sys.stderr, flush=True)

    print(f'Rendering Airy beam profile α={args.alpha}, λ={args.wavelength}Å => w₀={w0:.3g}Å, Rmax={rmax:.3g}Å, density={args.density}... ',
           file=sys.stderr, flush=True, end='')
    beam=AiryBeam(px=min(args.px_size)/args.density, r0=w0/(2*math.pi), rmax=rmax)
    print(' OK', file=sys.stderr, flush=True)

    try:
        psf = grid.fanout(beam, tilts=tilts, weighted=args.weighted)
    except KeyboardInterrupt:
        print("Exiting", file=sys.stderr)
        sys.exit(1)

    if args.crop:
        psf = autoshrink(psf)
        print(f'Fanout cropped to: {psf.shape[::-1]}')

    with mrcfile.new(args.output, overwrite=True) as mrc:
        mrc.set_data(psf)
        mrc.header.cella.x=args.px_size[0]*psf.shape[2]
        mrc.header.cella.y=args.px_size[1]*psf.shape[1]
        mrc.header.cella.z=args.px_size[2]*psf.shape[0]
        mrc.header['label'][0]=f'fancy: {label}'

if __name__ == "__main__":
    main()
