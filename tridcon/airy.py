# Copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3
# See the LICENSE-GPL-3 for details

import os
import threading
import concurrent.futures
import itertools
import numpy as np
import scipy

class Grid:
    """
    A grid class for spatial discretization with coordinate system conversion.
    This class represents a regular grid with specified shape and pixel size,
    providing coordinate-to-index conversion functionality.
    Parameters
    ----------
    shape : array-like
        Shape of the grid (number of pixels in each dimension).
    pxsize : array-like
        Pixel size in each dimension.
    dtype : numpy.dtype, optional
        Data type for the grid array (default: np.float32).
    eps : float, optional
        Tolerance for boundary checks (default: 1e-10).
    Attributes
    ----------
    shape : numpy.ndarray
        Grid dimensions.
    px : numpy.ndarray
        Pixel sizes for each dimension.
    borders : numpy.ndarray
        Half-extent of the grid in each dimension.
    boundaries : list
        Grid boundaries for each dimension.
    array : numpy.ndarray
        The actual grid data array.
    Methods
    -------
    x2i(coords)
        Convert spatial coordinates to grid indices.
    """
    def __init__(self, shape, pxsize, dtype=np.float32, ncpu=1):
        self.dtype=dtype
        self.shape=np.asarray(shape)
        self.px=np.asarray(pxsize, dtype=dtype)
        assert len(self.shape) == len(self.px), "Shape and pixel size must have the same number of dimensions"
        self.eps=10*self.px.max()**2/np.iinfo(np.uintp).max
        self.borders = self.px * self.shape / 2
        self.boundaries=[np.linspace(start=-b, stop=b, num=s+1) for b,s in zip(self.borders, self.shape)]
        self.array=np.zeros(self.shape, dtype=dtype)
        self.lock=threading.Lock()
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=ncpu)

    def direct_beam(self, beam, direction):
        """
        Generate a direct beam ray through the center of the grid.

        Args:
            beam: list of 2D point sources (u,v) and their weights
            direction: Direction vector as array-like

        Returns:
            list of 3D point sources (x,y,z) with weights
        """
        direction = np.asarray(direction, dtype=self.dtype)
        direction = direction / np.linalg.norm(direction)  # normalize
        
        # Create two orthogonal vectors perpendicular to direction
        # Find the axis with smallest component to avoid numerical issues
        min_axis = np.argmin(np.abs(direction))
        
        # Create first perpendicular vector
        u_vec = np.zeros_like(direction)
        u_vec[min_axis] = 1.0
        u_vec = u_vec - np.dot(u_vec, direction) * direction
        u_vec = u_vec / np.linalg.norm(u_vec)
        
        # Create second perpendicular vector using cross product
        v_vec = np.cross(direction, u_vec)
        
        sources = []
        for u,v,w in beam:
            pos = u * u_vec + v * v_vec
            sources.append([pos, w])
        
        return sources

    def ray(self, origin, weight, direction):
        ax = np.abs(direction).argmax()
        points = []
        for o,d,g in zip(origin, direction, self.boundaries):
            if d>-self.eps and d<self.eps:
                continue
            p=(direction[None,:] / d) * (g[:,None] - o) + origin[None,:]
            #mask = np.all(p >= -self.borders[None,:], axis=1) & np.all(p <= self.borders[None,:], axis=1)
            #p = p[mask]
            points.append(p)
    
        points=np.concatenate(points, axis=0)
        points=points[points[:,ax].argsort()]
        #points=points[np.unique(points[:,ax], return_index=True)[1]]
        midpoints=(points[1:]+points[:-1])/2
        lengths=np.linalg.norm(points[1:]-points[:-1], axis=1)
        ind = np.floor((midpoints + self.borders[None,:]) / self.px[None,:])
        mask = np.all((ind >= 0) & (ind < self.shape[None,:]), axis=1)

        ind = ind[mask].astype(np.intp)
        lengths = lengths[mask]
        with self.lock:
            self.array[*ind.T] += weight * lengths
        return

def _iterdens(it, i0, s):
    """
    Generate dense intervals by iterating at least as frequently as specified.

    Args:
        it: An iterable of numeric values (should be sorted in ascending order)
        i0: Starting value for interval generation
        s: Max step size for creating intervals

    Yields:
        tuple: (start, end) pairs representing intervals, where intervals are created
               with step size 's' until reaching each value in 'it', then from the
               last interval end to the current iteration value
    """
    for i in it:
        while i0+s < i:
            yield(i0,i0+s)
            i0+=s
        yield(i0,i)
        i0=i
    return

def _ring_unwrap(r, w, px):
    """
    Unwrap a ring into discrete point sources for wave simulation.

    Converts a circular ring of given radius into a series of point sources
    arranged around the circumference. For r=0, returns a single central source.

    Args:
        r (float): Radius of the ring
        w (float): Weight/amplitude of each source
        px (float): Pixel size for determining angular resolution

    Returns:
        numpy.ndarray: Array of shape (n, 3) where each row contains [x, y, weight]
                      coordinates of a point source
    """
    if 0==r:
        return(np.array([[0,0,w*2*np.pi]]))

    sources=[]
    angle0=np.random.random()*2*np.pi
    sectors=4+int(2*np.pi*r/px)
    angle1=2*np.pi/sectors
    w*=angle1
    r*=2*np.sin(angle1/2)/angle1
    for s in range(sectors):
        angle=angle0+s*angle1
        sources.append(np.array([r*np.cos(angle), r*np.sin(angle), w]))
    return(sources)

class Beam:
    def __init__(self, px, r0, rmax):
        np.random.seed(seed=0)
        self.px=px
        self.r0=r0
        self.rmax=rmax
        self.beamr= self.beam_radial()    
        self.beamuv = list(itertools.chain.from_iterable(_ring_unwrap(r, w, px) for r,w in self.beamr))

    def beam(self, grid, direction, weight=1.0, write=print):
        futures = []
        for s,w in grid.direct_beam(self.beamuv, direction):
            futures.append(grid.executor.submit(grid.ray, s, w*weight, direction))
        try:
            concurrent.futures.wait(futures)
        except KeyboardInterrupt:
            write("\033[A\033[KCaught KeyboardInterrupt, terminating workers")
            grid.executor.shutdown(wait=False, cancel_futures=True)
            raise

    def beam_radial(self):
        """
        Calculate beam intensity distribution using Airy disk pattern.
        This function computes the radial beam profile based on the first-order Bessel function
        squared, which represents the Airy disk intensity pattern. It integrates the beam
        intensity over radial segments and returns the radial positions and corresponding
        integrated intensities.
        Parameters
        ----------
        px : float
            Pixel size or sampling resolution
        r0 : float
            Characteristic radius parameter (Airy disk radius)
        rmax : float
            Maximum radius to calculate the beam profile
        Returns
        -------
        list of list
            List of [radius, integrated_intensity] pairs, where each pair represents
            the average radius and total integrated intensity for a radial segment
        """
        def f(r):
            x=r/self.r0
            return((scipy.special.j1(x)**2)/x)
        def fr(r):
            x=r/self.r0
            return(self.r0*(scipy.special.j1(x)**2))

        jzeros=list(self.r0*scipy.special.jn_zeros(1, int(10+self.rmax/(3*self.r0))))

        r=min(self.px, jzeros[0]/2)
        g=scipy.integrate.quad(f, 0, r)[0]
        beam=[[0,g]]

        for r1,r2 in _iterdens(jzeros, r, self.px):
            if r2>=self.rmax:
                break
            g=scipy.integrate.quad(f, r1, r2)[0]
            gr=scipy.integrate.quad(fr, r1, r2)[0]/g
            beam.append([gr,g])
        return(beam)
