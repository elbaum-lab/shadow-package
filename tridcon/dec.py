# Copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3
# See the LICENSE-GPL-3 for details

import time
import threading
import queue

import math
import numpy as np
import scipy

if __package__ is None or __package__ == '':
    from _cg import patch_scipy, MemoizeJac
    from _helpers import assrt, get_system_info
else:
    from ._cg import patch_scipy, MemoizeJac
    from ._helpers import assrt, get_system_info

patch_scipy()

import itertools
from collections import defaultdict

import torch

set_num_threads=torch.set_num_threads

emergency_stop=[]

class empty:
    def __getattr__(self, attr):
        return(self.__init__)
    def __setattr__(self, attr, value):
        return
    def __init__(self, *args, **kwargs):
        return

class SimpleQueue(queue.SimpleQueue):
    def set(self):
        self.put(None)

class Working:
    lock=threading.Lock()
    ret=None
    x=None
    objects=[]

    def __init__(self, device, jobs):
        self.device=device
        self.jobs=jobs
        self.objects.append(self)
        self.thread=None
        self.cpu = (device=='cpu')
        return

    def __del__(self):
        if self.thread:
            Working.x=None
            self.todo.set()
            self.thread.join()
            self.thread=None
            emergency_stop.remove(self.done)
            self.done=None
        return

    def runjobs(self, x):
        for j in self.jobs:
            r=list(j(x))
            with Working.lock:
                if Working.ret is None:
                    Working.ret=r
                else:
                    for i,v in enumerate(r):
                        Working.ret[i]+=v       # equivalent to Working.ret[i].add_(v) 

    def background(self):
        while True:
            self.todo.wait()
            self.todo.clear()
            if Working.x is None:
                break
            self.runjobs(Working.x)
            self.done.set()

    def foreground(self):
        if self.cpu:
            self.runjobs(Working.x)

    def submit(self):
        if self.cpu:
            return
        if self.thread is None:
            self.todo=threading.Event()
            self.done=threading.Event()
            emergency_stop.append(self.done)
            self.thread=threading.Thread(target=self.background)
            self.thread.start()
        self.todo.set()
        return()

    def wait(self, timeout=60):
        if not self.cpu:
            while not self.done.wait(timeout=timeout):
                if not self.thread.is_alive():
                    self.thread.join()
                    self.thread=None
                    raise RuntimeError('Worker thread died')
            self.done.clear()
        return

    @classmethod
    def func(cls, x):
        Working.x=x
        [o.submit() for o in cls.objects]
        [o.foreground() for o in cls.objects]
        [o.wait() for o in cls.objects]
        ret=cls.ret
        Working.x=None
        cls.ret=None       
        return(ret)

    @classmethod
    def clear(cls):
        for o in cls.objects:
            o.__del__()
    
def torchdev(d):
    try:
        return(str(torch.tensor([],device=d).device))
    except (RuntimeError, TypeError):
        return(d)

def _device_type(d):
    """Return the PyTorch device-type string ('cuda', 'mps', 'xpu', …) for device specifier d."""
    return torch.device(d).type

def _get_device_props(d):
    """Return a device-properties object/string, dispatched by device type."""
    t = _device_type(d)
    if t == 'cuda':             # covers CUDA and ROCm (HIP)
        return torch.cuda.get_device_properties(d)
    elif t == 'mps':            # Apple Metal
        return f'Apple MPS (driver_allocated={torch.mps.driver_allocated_memory()/1e9:.3g}GB)'
    elif t == 'xpu':            # Intel Arc / Ponte Vecchio
        return torch.xpu.get_device_properties(d)
    else:
        return f'{t} device'

def _device_mem_stat(d):
    """Return a memory-stats string for an accelerator device, dispatched by type."""
    t = _device_type(d)
    if t == 'cuda':             # covers CUDA and ROCm (HIP)
        alloc = torch.cuda.max_memory_allocated(d) / 1e9
        res   = torch.cuda.max_memory_reserved(d)   / 1e9
        total = torch.cuda.get_device_properties(d).total_memory / 1e9
        return f'{alloc:.3g}+{res:.3g}/{total:.3g}GB'
    elif t == 'mps':            # Apple Metal
        alloc  = torch.mps.current_allocated_memory() / 1e9
        driver = torch.mps.driver_allocated_memory()  / 1e9
        return f'{alloc:.3g}/{driver:.3g}GB'
    elif t == 'xpu':            # Intel Arc / Ponte Vecchio
        alloc = torch.xpu.max_memory_allocated(d) / 1e9
        res   = torch.xpu.max_memory_reserved(d)   / 1e9
        total = torch.xpu.get_device_properties(d).total_memory / 1e9
        return f'{alloc:.3g}+{res:.3g}/{total:.3g}GB'
    else:
        return 'n/a'

def argandmin(l):
    return(min(enumerate(l), key=lambda x: x[1]))
def argandmax(l):
    return(max(enumerate(l), key=lambda x: x[1]))

def sum_sqr(x):
    return(x.square().sum())
    #x=x.reshape(-1)
    #return(torch.dot(x,x))

def rfft(input, n=None, dim=-1, norm=None, *, out=None):
    assert -1 == dim, f'Only dim=-1 is supported for rfft, got {dim=}'

    if n is not None and n > input.shape[-1]:
        value = input.mean().item()
        pad = n - input.shape[-1]
        input = torch.nn.functional.pad(input, (0, pad), mode='constant', value=value)

    return(torch.fft.rfft(input, n=n, dim=dim, norm=norm, out=out))

def resize_fft_input(x, dims, sizes):
    assert len(dims) == len(sizes), f'resize_fft_input: len({dims}) != len({sizes})'
    xshape = x.shape
    must_pad = False
    pad_amount = [0] * len(xshape) * 2                  # (pad_left, pad_right) for each dimension
    for i, (d, s) in enumerate(zip(dims, sizes)):
        if s > xshape[d]:
            must_pad = True
            p = s - xshape[d]
            pi = len(pad_amount) - 1 - 2 * d
            pad_amount[pi] = p                          # pad on the right side

    if must_pad:
        value = x.mean()
        if value.is_complex():
            x = torch.complex(torch.nn.functional.pad(x.real, pad_amount, mode='constant', value=value.real),
                              torch.nn.functional.pad(x.imag, pad_amount, mode='constant', value=value.imag))
        else:
            x = torch.nn.functional.pad(x, pad_amount, mode='constant', value=value)

    return x

def rfftn(input, s=None, dim=None, norm=None, *, out=None):
    if s is not None:
        input = resize_fft_input(input, dims=dim, sizes=s)
    return(torch.fft.rfftn(input, s=s, dim=dim, norm=norm, out=out))

def fftn(input, s=None, dim=None, norm=None, *, out=None):
    if s is not None:
        input = resize_fft_input(input, dims=dim, sizes=s)
    return(torch.fft.fftn(input, s=s, dim=dim, norm=norm, out=out))

class convolver:
    @staticmethod
    def autoshrink(a):
        tight=[]
        for i in range(a.ndim):
            b=np.swapaxes(a,i,0)
            t=True
            while b.shape[0]>2 and not np.any(b[0]) and not np.any(b[-1]):
                b=b[1:-1]
                t=False
            tight.append(t)
            a=np.swapaxes(b,i,0)

        return(tight, a)

    @staticmethod
    def kernel_shift(k, fftaxes):
        shift = [-((dim-1)//2) for dim in np.asarray(k.shape)[fftaxes]]
        return(torch.roll(k, shift, dims=fftaxes))

    @staticmethod
    def deprime(n, primes=[2,3,5,7], maxpad=10):
        pmin=n
        nx=n
        for i in range(n,n+maxpad):
            k=i
            for p in primes:
                while 0==k%p:
                    k//=p
            if k<pmin:
                pmin=k
                nx=i
            if pmin==1:
                break
        return(nx)

    @staticmethod
    def fft_slider(src, kf, shape, fftaxes, forward):
        def warray():
            w=[torch.empty_like(k) for k in kf]
            i=0
            while True:
                yield(w[i])
                i+=1
                i%=length
        def fft_iter(src, shape, fftaxes):
            for x,o in zip(src,warray()):
                yield(rfftn(x.to(device), s=shape, dim=fftaxes, out=o))

        length=len(kf)
        device=kf[0].device
        w=[]
        if forward:
            head_length=length//2
        else:
            head_length=(length-1)//2
        head=[None]*head_length
        tail=[None]*(length-1-head_length)
        for x in itertools.chain(head, fft_iter(src, shape, fftaxes), tail):
            w.append(x)
            if len(w)==length:
                yield(w)
                w.pop(0)
        return

    def describe(self):
        s=f'tight={self.tight}, fftaxes={self.fftaxes}, dir0={self.dir0}, moveaxis={self.moveaxis}, vshape={self.vshape}, '
        s+=f'kernel{self.kernelshape}, kf{tuple(self.kf.shape)}, shape={self.shape}, crop={self.crop}'
        return(s)

    def testfit(self, volume, low=.5, high=0.999):
        """
        Test the fit quality comparing overlapping low-magnitude spatial harmonics of volume and kernel.
        """
        assrt(self.semi_fft is None, 'Can testfit only with full FFT convolver')
        assrt(self.dir0 is False, 'Can testfit only with non-directional convolver')

        vf=rfftn(volume, s=self.shape, dim=self.fftaxes)
        vf_shape=vf.shape
        vf=(vf.flatten().abs()**2).numpy(force=True)

        self.kf = (self.kf.expand(vf_shape).flatten().abs()**2).numpy(force=True)
        klow, khigh = np.quantile(self.kf, [low, high])
        self.kf = self.kf < klow
       
        assrt(vf.shape==self.kf.shape, f'Shape mismatch: {vf.shape}!={self.kf.shape}')

        vlow=np.quantile(vf,high)*klow/khigh
        vf = vf < vlow

        both=(vf&self.kf).sum()/(low*self.kf.size)
        return(both - vf.sum()/vf.size)

    def __init__(self, kernel, vshape, fft_thrsh=1, semi_fft=False, device='cpu'):
        assrt(len(vshape)==kernel.ndim, f'Can not convolve {vshape}*{kernel.shape}')

        tight, kernel = self.autoshrink(kernel)
        self.kernelshape=kernel.shape

        # check if the kernel size is sufficient
        for (t,k,v) in zip(tight, kernel.shape, vshape):
            s=2*v
            if t and k<s:
                raise ValueError(f'Tight kernel {kernel.shape} is too small for {vshape}: {k}<{s}')

        kernel=torch.asarray(kernel)

        # further shrink kernel if larger than needed
        for i,(k,v) in enumerate(zip(kernel.shape, vshape)):
            s=2*v
            if k>s:
                kernel=torch.swapaxes(kernel, 0, i)
                head=(k+1)//2-v
                kernel=kernel[head:head+s]
                kernel=torch.swapaxes(kernel, 0, i)

        # compute fftaxes and set dir0/dir_axis
        a,m=argandmin(kernel.shape)
        if m<=max(1, fft_thrsh):
            semi_fft=False
        fftaxes=[]
        self.dir0=False
        for i,k in enumerate(kernel.shape):
            if a==i and k<=fft_thrsh:
                kernel=torch.flip(kernel, dims=(a,))
                self.dir0=True
                dir_axis=a
            elif k>1:
                fftaxes.append(i)
        assrt(len(fftaxes)>0, f'Kronecker kernel {kernel.shape}')

        # pad kernel as needed
        shape=[]
        for i,k in enumerate(kernel.shape):
            if i in fftaxes:
                v=vshape[i]
                s=min(2*v, v+k//2)
                s=2*self.deprime((1+s)//2)
                shape.append(s)
                if s>k:
                    head=(s-k)//2
                    kernel=torch.swapaxes(kernel, -1, i)
                    kernel=torch.nn.functional.pad(kernel, (head, s-k-head))
                    kernel=torch.swapaxes(kernel, -1, i)

        # reorder axes
        a,m=argandmax(vshape)
        if self.dir0:
            if dir_axis == a:
                axes=[dir_axis]
                axes_to=[0]
                self.long_axis=0
            else:
                axes=[dir_axis, a]
                axes_to=[0,1]
                self.long_axis=1
        else:
            self.long_axis=0
            axes=[a]
            axes_to=[0]

        # options here:
        # (a) k == 1 -> semi_fft=False
        # (b) k <= fft_thrsh -> dir0=True, semi_fft=False
        # (c) k == s  -> fftaxes
        if semi_fft:     # -> fftaxes is full
            semi_axis=min(((i,k) for i,k in enumerate(kernel.shape) if i not in axes), key=lambda x: x[1])[0] # find the shortest axis not in axes
            axes+=[semi_axis]
            axes_to+=[kernel.ndim-1]
            self.semi_fft=semi_fft
        else:
            self.semi_fft=None

        axes+=[i for i in range(kernel.ndim) if i not in axes]
        axes_to+=[i for i in range(kernel.ndim) if i not in axes_to]
        self.moveaxis=list(zip(*sorted(zip(axes, axes_to))))
        
        kernel=self.move(kernel, forward=True)

        _reordered=sorted((self.moveaxis[1][a],s) for a,s in zip(fftaxes, shape))
        self.fftaxes, self.shape=zip(*_reordered)
        self.fftaxes=list(self.fftaxes)

        self.tight=self.permute(tight)
        self.vshape=tuple(self.permute(vshape))
        self.crop=tuple(slice(s) for s in self.vshape)

        kernel=self.kernel_shift(kernel, self.fftaxes)
        self.kf=rfftn(kernel, dim=self.fftaxes).to(device)

        if self.dir0:
            self.fftaxes=[a-1 for a in self.fftaxes]

        return

    def permute(self, l):
        o=[None]*len(l)
        for a,t in zip(*self.moveaxis):
            o[t]=l[a]
        return(o)

    def move(self, volume, forward):
        if forward:
            volume=torch.moveaxis(volume,*self.moveaxis)
        else:
            volume=torch.moveaxis(volume,*self.moveaxis[::-1])
        return(volume)

    def xward(self, volume, forward):
        assert self.vshape==volume.shape, f'Volume shape mismatch {self.vshape}!={volume.shape}'
        assert self.kf.real.dtype==volume.dtype, f'Volume dtype mismatch {self.kf.real.dtype}!={volume.dtype}'      # see also numpy.result_type
        pin_memory = self.kf.device.type == 'cuda'   # pinned host memory is CUDA/ROCm only

        if forward:
            kf=self.kf
        else:
            kf=self.kf.conj()

        if self.dir0:
            crop=self.crop[1:]
            if forward:
                kf=kf.unbind(0)
            else:
                kf=kf.unbind(0)[::-1]
            vf=torch.empty_like(volume, pin_memory=pin_memory)
            for (i,o) in zip(self.fft_slider(volume, kf=kf, shape=self.shape, fftaxes=self.fftaxes, forward=forward), vf):
                acc=None
                for x,k in zip(i,kf):
                    if x is None:
                        continue
                    if acc is None:
                        acc=x*k
                    else:
                        torch.addcmul(acc, x, k, out=acc)     #acc+=x*k

                o[...]=torch.fft.irfftn(acc, s=self.shape, dim=self.fftaxes)[crop]
        else:
            if self.semi_fft is None:
                vf=rfftn(volume.to(kf.device), s=self.shape, dim=self.fftaxes)
                out=torch.empty_like(volume, pin_memory=pin_memory)
                vf*=kf
                out[...]=torch.fft.irfftn(vf, s=self.shape, dim=self.fftaxes)[self.crop]
                return(out)
            else:
                fftaxes=[a for a in self.fftaxes if a!=volume.ndim-1]
                shape=self.shape[:-1]
                vf=rfft(volume.to(device=self.semi_fft), n=self.shape[-1], dim=-1)
                for i in range(vf.shape[-1]):
                    vi=fftn(vf[...,i].to(kf.device), s=shape, dim=fftaxes)
                    vi*=kf[...,i]
                    vf[...,i]=torch.fft.ifftn(vi, s=shape, dim=fftaxes)[self.crop[:-1]]
                vf=torch.fft.irfft(vf, n=self.shape[-1], dim=-1)[self.crop].to(volume.device)
        return(vf)

    def forward(self, volume):
        return(self.xward(volume, forward=True))

    def backward(self, volume):
        return(self.xward(volume, forward=False))

class D2:
    def __init__(self, length=1.):
        self.length=np.asarray(length)
        self.dim=None
        return
    
    # returns diagonal elements of Hessian, sqr(2)*off-diagonal elements, laplacian
    def forward(self, a):
        if self.dim!=a.ndim:
            self.dim=a.ndim
            self.length=np.broadcast_to(self.length, self.dim)
            self.crop=[slice(1,-1),]*self.dim
            self.stencilsd=self.StencilsDiag(self.dim, 1./self.length)
            self.stencilsod=self.StencilsOffDiag(self.dim, (2.**(-3/4))/self.length)
            self.laplacian=self.StencilsAdd(self.stencilsd)           
            self.nd=len(self.stencilsd)
            self.nod=len(self.stencilsod)
        o=a.new_zeros(self.nd+self.nod+1, *np.array(a.shape)+2)
        ox=o[:,*(slice(2,-2),)*self.dim]

        for i,d in enumerate((*self.stencilsd, *self.stencilsod)):
            for j,(s,w) in enumerate(d):
                if j:
                    ox[i].add_(a[*s], alpha=w)                      # ox[i]+=w*a[*s]
                else:
                    torch.multiply(w, a[*s], out=ox[i])

        ox[-1]=ox[:self.nd].sum(dim=0)    #Laplacian
        return(o)        

    def backward(self,v):
        assert self.dim==v.ndim-1, f'backward: Input shape mismatch: expected {self.dim+1} dimensions but got {v.ndim}'

        o=torch.zeros_like(v)
        ox=o[:,*(slice(1,-1),)*self.dim]

        for i,d in enumerate((*self.stencilsd, *self.stencilsod, self.laplacian)):
            for j,(s,w) in enumerate(d):
                if j:
                    ox[i].add_(v[i,*s], alpha=w)                     #ox[i]+=w*v[i,*s]
                else:
                    torch.multiply(w, v[i,*s], out=ox[i])
        return(o)

    @staticmethod
    def StencilsDiag(d, r):
        ret=[
            [[[slice(1,-1),]*d,-2*r[i]**2],
             [[slice(1,-1),]*d,r[i]**2],
             [[slice(1,-1),]*d,r[i]**2]
            ] for i in range(d)]

        for i in range(d):
            ret[i][1][0][i]=slice(2,None)
            ret[i][2][0][i]=slice(None,-2)
        return(ret)

    @staticmethod
    def StencilsOffDiag(d, r):
        ret=[
            [[[slice(1,-1),]*d,None],
             [[slice(1,-1),]*d,None],
             [[slice(1,-1),]*d,None],
             [[slice(1,-1),]*d,None]
            ] for _ in range(d*(d-1)//2)]

        for i,(j,k) in enumerate(np.nditer(np.triu_indices(d,1))):
            ret[i][0][0][k]=slice(2,None)
            ret[i][0][0][j]=slice(2,None)
            ret[i][0][1]=r[k]*r[j]

            ret[i][1][0][k]=slice(None,-2)
            ret[i][1][0][j]=slice(None,-2)
            ret[i][1][1]=r[k]*r[j]
            
            ret[i][2][0][k]=slice(None,-2)
            ret[i][2][0][j]=slice(2,None)
            ret[i][2][1]=-r[k]*r[j]
    
            ret[i][3][0][k]=slice(2,None)
            ret[i][3][0][j]=slice(None,-2)
            ret[i][3][1]=-r[k]*r[j]
        return(ret)

    @staticmethod
    def StencilsAdd(stencils):
        sum={}
        for oper in stencils:
            for s in oper:
                strs=repr(s[0])
                if strs in sum:
                    sum[strs][1]+=s[1]
                else:
                    sum[strs]=s.copy()
        return(list(sum.values()))    

# Smoothness regularization
# alpha1: weight of sum of squares of second derivatives
# alpha2: weight of squared laplacian
class J1:
    def __init__(self, length, lamb, alpha1, alpha2, split_chunks, dim, dev='cpu', eps=1e-5):
        self.d=D2(length=length)
        self.dev=dev
        self.pin_memory = _device_type(torchdev(dev)) == 'cuda'   # pinned host memory is CUDA/ROCm only
        self.lamb=lamb
        self.alpha1=alpha1
        self.alpha2=alpha2
        self.eps=max(eps, np.finfo(np.float32).tiny)
        self.split_chunks_J1=split_chunks
        self.dim=dim
        return

    # from left to right along dim
    def part(self, x, dim, left, right):
        crop1=[slice(1,-1)]*x.ndim
        crop=crop1.copy()
        crop[dim]=slice(left+1,right+1)

        Eg=self.eps*x.new_ones(*np.array(x.shape)+2)
        Eg[*crop1]+=x*x
        d2=self.d.forward(x)
        for d in d2[:-1]:
            Eg+=self.alpha1*d*d                      # sum of all squares
        Eg+=self.alpha2*d2[-1]**2                    # squared laplacian
        H=torch.log(Eg[*crop]).sum()*self.lamb/2

        d2/=Eg[None,...]
        d2=self.d.backward(d2)
        grad=self.alpha2*d2[-1,*crop]
        for d in d2[:-1]:
            grad+=self.alpha1*d[*crop]
        grad+=self.cropper(x, dim, slice(left,right))/Eg[*crop]
        grad*=self.lamb
        return(H,grad)

    # We should split the longest direction L into chunks of size up to ms
    # so as to keep width-2 margins (second derivative^2)
    def splitter(self, L):
        if not self.split_chunks_J1:
            yield(slice(0,L), (0,L), slice(0,L))
            return

        max_J1_size=max(L//self.split_chunks_J1, 6)

        left=0
        lv=0
        while True:
            right=left+max_J1_size
            if right>=L:
                src=slice(left,L)
                select=(lv, L-left)
                dst=slice(left+lv, L)
                yield(src, select, dst)
                break
            else:
                src=slice(left, right)
                select=(lv, max_J1_size-2)
                dst=slice(left+lv, right-2)
                yield(src, select, dst)
                left=right-4
                lv=2
        return

    @staticmethod
    def cropper(x, dim, s, source=None):
        crop=[slice(None)]*x.ndim
        crop[dim]=s
        if source is None:
            return(x[*crop])
        else:
            x[*crop]=source
    
    def __call__(self, x):
        L=x.shape[self.dim]
        H=None
        grad=torch.empty_like(x, pin_memory=self.pin_memory)
        for src, (left,right), dst in self.splitter(L):
            h,g=self.part(self.cropper(x, self.dim, src).to(self.dev), self.dim, left, right)
            if H is None:
                H=h
            else:
                H+=h
            self.cropper(grad, self.dim, dst, g)
        return(H.to(grad),grad)


class tomogram:
    def __init__(self, raw, psf, intermediate, result=None, lamb=1e-4, lampos=None, alpha1=1., alpha2=1., eps=1e-5,
                 logging=False, maxrss=False, checkpoint=10,
                 split_chunks_J1=None, fft_thrsh=1, fft_mem=False,
                 dev_con='cpu', dev_reg='cpu', parallel=True,
                 length=None, dry_run=False):

        if length is None:
            length=[1.]*raw.ndim
        if lampos is None:
            lampos=1000
        self.nit=0
        self.evals=0
        self.lampos=lampos*lamb
        self.checkpoint=checkpoint
        self.dirty=False
        self.overflow=False
        self.intermediate=intermediate

        if dry_run:
            fft_thrsh=1
            fft_mem=False

        self.con=convolver(kernel=psf, vshape=raw.shape, fft_thrsh=fft_thrsh, semi_fft=fft_mem, device=dev_con)

        raw=self.con.move(torch.asarray(raw), forward=True).clone()
        self.raw=raw
        self.loss_factor=10000/raw.numel()

        if not dry_run:
            if result is None:
                self.result=raw
            else:
                self.result=self.con.move(torch.asarray(result), forward=True)

        self.J1=J1(length=self.con.permute(length), dev=dev_reg, lamb=lamb, alpha1=alpha1, alpha2=alpha2, split_chunks=split_chunks_J1, dim=self.con.long_axis, eps=eps)
        
        self.shape=self.raw.shape
        self.start_time=time.time()

        self.func=MemoizeJac(self.objective)
        self.jac=self.func.derivative

        self.logging=logging
        self.maxrss=maxrss
        self.log(f'Tomogram init(raw{raw.shape}, psf{psf.shape}, result{'=None' if result is None else result.shape}, {lamb=}, {lampos=}, {alpha1=}, {alpha2=}, {eps=}, {checkpoint=}, {split_chunks_J1=}, {length=})')

        for line in get_system_info(NumPy=np.__version__, PyTorch=torch.__version__):
            self.log(line)

        fft_mem=torchdev(fft_mem)
        dev_con=torchdev(dev_con)
        dev_reg=torchdev(dev_reg)
        self.accel_devices={fft_mem, dev_con, dev_reg} - {False, 'cpu'}
        for d in self.accel_devices:
            try:
                self.log(f'{d}: {_get_device_props(d)}')
            except(AssertionError,RuntimeError) as e:
                e.add_note(f'{d} is unavailable')
                raise
            except ValueError:
                pass
        if parallel:
            parallel=defaultdict(list)
            parallel[dev_con].append(self.J0)
            parallel[dev_reg].append(self.J1)
            parallel['cpu'].append(self.J2)
            for device,j in parallel.items():
                Working(device, j)
        self.parallel=parallel
        self.log(f'{parallel=}')
        
        self.log('Convolver('+self.con.describe()+')')
        return

    def log(self, message, *args, **kwargs):
        if self.logging:
            self.logging(f'[{time.time()-self.start_time:>7.1f}]:{message}', *args, **kwargs)

    # Deconvolution error
    def J0(self, x):
        y=self.con.forward(x)-self.raw
        loss=sum_sqr(y)
        y=2*self.con.backward(y)
        return(loss,y)        

    
    # Negativity penalty
    def J2(self,x):
        grad=torch.clip(x, min=None, max=0)
        loss=sum_sqr(grad)/2
        return(loss*self.lampos,grad*self.lampos)

    def objective(self, x):
        self.evals+=1
        x=torch.asarray(x)
        x=x.reshape(self.shape)

        if self.parallel:
            loss,grad=Working.func(x)
        else:
            loss,grad=self.J0(x)

            l,g=self.J1(x)
            loss+=l
            grad+=g

            l,g=self.J2(x)
            loss+=l
            grad+=g

        grad=grad.reshape(-1)
        # The following clips NaNs but slows down the optimization
        # grad.nan_to_num_()
        return(loss,grad)

    def debug(self):
        global rfft, rfftn, fftn
        rfft=torch.fft.rfft
        rfftn=torch.fft.rfftn
        fftn=torch.fft.fftn

        loss,grad=self.objective(self.result)
        return(loss,self.con.move(grad.reshape(self.shape), forward=False).numpy())
    
    def copy_result(self, result):
        self.result=torch.asarray(result.x).reshape(self.shape)
        self.dirty=True
        return

    def memstat(self):
        s=f'cpu:{self.maxrss()}'
        for d in self.accel_devices:
            s+=f', {d}:{_device_mem_stat(d)}'
        return(s)
      
    def dryrun(self, arg=''):
        s='Parameters valid.\n'
        if arg=='psf':
            s+=f'PSF fitness is {self.con.testfit(self.raw)*100:.3g}%.\n'
        s+=self.memstat()
        self.log(f'Dry run: {s}')
        return(s)

    def optimize(self, maxiter=100, save_callback=None, pbar=empty(), verbose=False):
        def postfix(l):
            post=f'loss: {l*self.loss_factor:.3g}'
            if verbose:
                post+=f', {self.memstat()}'
            return(post)

        def callback(intermediate_result: scipy.optimize.OptimizeResult):
            l=intermediate_result.fun
            self.nit+=1
            self.log(f'callback[{self.nit:02}.{self.evals:03d}]: {l}, {self.memstat()}')

            if redbutton.is_set():
                self.copy_result(intermediate_result)
                raise StopIteration
            else:
                pbar.set_postfix_str(postfix(l), refresh=False)
                pbar.update()
                self.copy_result(intermediate_result)

                if self.intermediate(self.nit):
                    self.log(f'Intermediate iteration {self.nit} reached')
                    minimize_news.put('intermediate')
            return

        def minimize():
            self.OptimizeResult=scipy.optimize.minimize(self.func,
                                                        self.result.reshape(-1),
                                                        callback=callback,
                                                        method='CG',
                                                        jac=self.jac,
                                                        options={'maxiter':maxiter, 'disp':False})
            self.nit=self.OptimizeResult.nit
            self.copy_result(self.OptimizeResult)
            if not redbutton.is_set():
                pbar.total=pbar.n
                self.log(f'Optimization terminated {"" if self.OptimizeResult.success else "un"}sucessfully [{self.OptimizeResult.status}] due to {self.OptimizeResult.message}')

            if not np.isfinite(self.OptimizeResult.jac).all():
                self.log('Warning: Overflow detected in gradient!')
                self.overflow=True
            if not np.isfinite(self.OptimizeResult.fun).all():
                self.log('Warning: Overflow detected in loss!')
                self.overflow=True

            pbar.refresh()
            minimize_news.set()
            return

        def save():
            if save_callback:
                if self.dirty:
                    self.log('Saving output')
                    self.dirty=False
                    save_callback(self.con.move(self.result, forward=False).numpy(), iteration=self.nit, pbar=pbar)
                else:
                    self.log('No changes to save')
            return

        def excepthook(args):
            self.log(f"Uncaught exception in {args.thread}", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
            exc_name = args.exc_type.__name__
            exc_msg = str(args.exc_value) if args.exc_value else "No details available"
            parts = exc_msg[:120].split('. ', 3)
            exc_msg = '. '.join(parts[:3]) + '.'
            pbar.write(f"{exc_name}: {exc_msg}")
            for b in emergency_stop:
                b.set()
            return

        redbutton=threading.Event()
        emergency_stop.append(redbutton)

        minimize_news=SimpleQueue()
        emergency_stop.append(minimize_news)

        pbar.set_postfix_str(postfix(math.nan), refresh=True)
        self.log(f'optimize: {self.memstat()}')

        threading.excepthook=excepthook
        worker=threading.Thread(target=minimize)
        worker.start()

        while worker.is_alive():
            try:
                news=minimize_news.get(timeout=self.checkpoint)
                if news=='intermediate':
                    save()
                else:
                    assert news is None, f'Unexpected minimize_news value: {news}'
                    break
            except queue.Empty:             # timeout expired, checkpoint reached
                self.log(f'Checkpoint reached')
                save()

            except KeyboardInterrupt:
                self.log('Catched Ctrl-C')
                pbar.write('\033[A\033[KCleaning up, please wait...')
                redbutton.set()
        self.log('Joining worker thread')
        worker.join()
        self.log('Saving optimization result')
        Working.clear()
        save()
        if self.overflow:
            pbar.warn('numerical overflow detected!')
        return

