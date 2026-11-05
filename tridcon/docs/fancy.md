# Fanout generation tool (`fancy`)

## Usage

```sh
fancy [--help] --output OUTPUT --wavelength WAVELENGTH --alpha ALPHA ( --ref TOM | --shape N N N --px-size P,P,P ) [options]
```

### Required arguments

- `--output OUTPUT`  
  Output MRC file name.

- `--wavelength λ`  
  Beam wavelength in Angstroms.

- `--alpha α`  
  Beam semi-divergence angle in radians.

### Output volume specification (either `--ref` or both `--shape` and `--pxsize` must be provided)

- `--shape N N N`
  Shape \[x,y,z\] of the grid in voxels.

- `--px-size P,P,P`
  Voxel dimensions \[x,y,z\] in Angstroms.

- `--ref TOM`
  Use the geometry of the tomogram as reference

### Optional arguments

- `--tilt-axis {x,y,z}`
  Axis about which to tilt (default: `y`).

- `--neg-angle`  
  Negate tilt angles (multiply by -1).

- `--base-angle θ₀`
  Tilt angles offset in degrees (default: `0`).

- `--tilts TILTS`
  Path to a text file with the tilts information, or `-` (default) to read from standard input.

- `--rabs R`
  Absolute radius for the PSF mask in Angstroms (default: use --rrel value below).

- `--rrel r`
  Relative radius for the PSF mask in units of the beam waist (default: `10.0`).

- `--density d`
  Subpixel density for the beam rendering (default: `4.0`).

- `--weighted`
  Weight the beam intensity by the cosine of the tilt angle.

- `--crop`
  Crop the PSF to non-zero region before saving.

### Tilt file format
Empty lines and lines starting with `#` are ignored. All angles are in degrees. Related options: `--tilt-axis`, `--neg-angle`, `--base-angle`.
Each line in the tilt file can have either one or two angles:

- One angle: `--tilt-axis` specifies the axis of rotation.

- Two angles: tilt and azimuth (polar θ and azimuthal φ angles). Options `--tilt-axis` and `--base-angle` are ignored.

### Example:

Create a file `tilts.txt`:

```
-45
-30
0
30
60
```

or

```
-34
56
1.23    88.88
45.67   88.88
```

Run:

```sh
fancy --output fanout.mrc --shape 2049 67 345 --px-size 5 5 5 --wavelength 2.0 --alpha 1e-3 --density 2.0 --rrel 20 --tilt-axis x --tilts tilts.txt --crop
```

Combine two tilt files `t1.tlt` and `t2.tlt` for dual-axis operations:

```sh
angle2=-89.5
( cat t1.tlt ; awk -v a=$angle2 'NF {$0 = $0 " " a} 1' t2.tlt ) | fancy --output dual.mrc --ref tomogram.mrc --wavelength 0.002 --alpha 1e-3
```

## Output

The output is a 3D MRC file containing the summed Airy disk PSF fanout for the specified tilts and parameters. The MRC header is annotated with the PSF and tilt information.

## Airy disk beam profile

$$
\text{PSF}_\text{Airy}(r) = \left(\frac{w_0}{\pi r}J_1\left(\frac{2\pi r}{w_0}\right)\right)^2,
$$

where $J_1$ is the first-order Bessel function of the first kind, and

$$
w_0 = \frac{\lambda}{\alpha}.
$$

