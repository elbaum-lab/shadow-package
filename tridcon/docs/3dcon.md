# Deconvolution tool (`3dcon`)

This program performs tomographic deconvolution based on the method of [Arigovindan et al.][1]

---

## Usage

```sh
3dcon [--help] [--config 3dcon.conf] [options] <infile>
```

| Argument                    | Type            | Required | Description                                                                                  |
|-----------------------------|-----------------|----------|----------------------------------------------------------------------------------------------|
| `<infile>`                  | File name       | No       | Input MRC file with reconstructed tomogramm for deconvolution (default: `stdin`)             |
| `--psf <file>`              | File name       | Yes      | PSF file (MRC format)                                                                        |
| `--output <file>`           | File name       | Yes      | Output MRC file name (can use formatting, see below)                                         |
| `--label <label>`           | String          | No       | Label for MRC header (can use formatting, see below)                                         |
| `--thumbnail <file>`        | String          | No       | Save thumbnail PNG of the central slice (supports format placeholders)                       |
| `--thmb-axis {x,y,z}`       | String          | No       | Axis for thumbnail slice (default: z)                                                        |
| `--reuse`                   | Flag            | No       | Reuse existing output if present                                                             |
| `--negative`                | Flag            | No       | Assume negated input/output                                                                  |
| `--flip {x,y,z}`            | String          | No       | Flip PSF over axis                                                                           |
| `--2D {x,y,z}`              | String          | No       | Squeeze axis in PSF for 2D convolution                                                       |
| `--psf-tola <tolerance>`    | Float           | No       | Absolute tolerance for PSF autoshrink (default: 0)                                           |
| `--psf-tolr <tolerance>`    | Float           | No       | Relative tolerance for PSF autoshrink (default: 0)                                           |
| `--autosave <sec>`          | Integer         | No       | Checkpoint interval in seconds                                                               |
| `--intermediate <spec>`     | String          | No       | Comma-separated list of iterations to save (e.g., "5,20,70" or ":10" for every 10th)         |
| `--iterations <N>`          | Integer         | Yes      | Maximum number of optimization iterations                                                    |
| `--epsilon <` $\epsilon$ `>`| Float           | No       | A small number to regularize log (one shouldn't have to touch it)                            |
| `--lambda <` $\lambda$ `>`  | Float           | Yes      | Regularization strength                                                                      |
| `--px-size <x,y,z>`         | Floats          | No       | Override MRC pixel size, Angstroms (comma-separated, e.g. 1.0,1.0,1.0)                       |
| `--unit <px/angstrom>`      | String          | No       | Units for the lengths $l_A, l_L$ (default: px)                                               |
| `--lenA <` $l_A$ `>`        | Float           | Yes      | Arigovindan's smoothing length                                                               |
| `--lenL <` $l_L$ `>`        | Float           | Yes      | Laplacian smoothing length                                                                   |
| `--lampos <` $\lambda_+$ `>`| Float           | Yes      | Negativity penalty                                                                           |
| `--parallel`                | Flag            | No       | Run on available devices in parallel                                                         |
| `--cpu <N>`                 | Integer         | No       | Number of CPUs to use (default: all available)                                               |
| `--dev-reg <dev>`           | String          | No       | Device for regularization (default: `cpu`)                                                   |
| `--dev-con <dev>`           | String          | No       | Device for convolutions (default: `cpu`)                                                     |
| `--fft-mem <dev>`           | String          | No       | Optimize FFT for device memory (default: `none`)                                             |
| `--fft-thrsh <N>`           | Integer         | No       | Threshold PSF thickness for FFT convolve (default: 3, 0 to disable)                          |
| `--mem-guard [LIMIT]`       | Integer         | No       | Install an out-of-memory (OOM) guardrail (detects automatically if argument not supplied)    |
| `--memory [N]`              | Integer         | No       | Optimize for memory (regularization term $J_1$ split into N chunks, 9 if not specified)      |
| `--quiet`                   | Flag            | No       | Disable progress bar                                                                         |
| `--verbose`                 | Flag            | No       | Verbose progress bar                                                                         |
| `--dry-run [psf]`           | String          | No       | Check parameters and exit (specify `psf` as an argument to compute psf fitness)              |

---

## Configuration File

You can specify options in a config file (default: `3dcon.conf`). Each line should be in the format:

```
option=value
```

Lines starting with `#` are comments. Command-line options override config file options.

---

## Output File Name / MRC Label Formatting

The output file names (both `--output` and `--thumbnail`) and `--label` can include placeholders, which are replaced at runtime:

- `{version}`: 3dcon package version
- `{lambda}`: Regularization parameter
- `{lenL}`: Laplacian smoothing length
- `{lenA}`: Arigovindan smoothing length
- `{unit}`: Unit [A or P] (Angstrom or pixel)
- `{iteration}`: Iteration number
- `{utc}`: Timestamp
- `{negative}`: N/n flag for negated input
- `{flip}`: Axis (x/y/z) or '-' if none
- `{epsilon}`: Epsilon value
- `{lampos}`: Lambda+ (negativity penalty) value  
- `{2D}`: 2D axis (x/y/z) or '-' if none
- `{psf}`: PSF filename
- `{infile}`: Input file name

Example:  
`output=deconv_{version}_{iteration}.{utc}.mrc`

---

## Parallelism

- Use `--parallel` to enable parallel computation across devices.
- Use `--cpu N` to set the number of CPU threads *per parallel computation goal*.

---

## Device names

- `--dev-reg`, `--dev-con`, and `--fft-mem` accept pytorch device names (e.g., `cpu`, `cuda`, `cuda:0`, `xpu`, `mps`).

---

## Checkpointing

- Use `--autosave <seconds>` to save intermediate results periodically.
- Use `--intermediate <spec>` to save specific iterations:
  - Specify exact iterations: `--intermediate "5,20,70"` saves iterations 5, 20, and 70
  - Specify intervals with colon (or slash `/`) prefix: `--intermediate ":10"` saves every 10th iteration (10, 20, 30, ...)
  - Combine both: `--intermediate "4,8,16,32,:10"` saves iterations 4, 8, 10, 16, 20, 32, 40, 50 etc
- To keep individual checkpoint files, use `{iteration}` or/and `{utc}` in the `--output` name. Otherwise, the checkpoint will overwrite the previous one.
- If interrupted (Ctrl-C), the program will clean up and save the current result.

---


## Logging

- Logs are written to `3dcon.log` in the current directory.
- Progress is shown via a progress bar unless `--quiet` is specified.

---

## Troubleshooting

- **Host Out of Memory Events (ulimit etc)**
- **GPU Out of Memory (cuda)**
- **Input/Output shape mismatch**: Ensure your input and PSF files are compatible.
- **Output file exists**: Use `--reuse` to overwrite or reuse existing output.
- **Pixel size mismatch**: Use `--px-size` to override MRC header values if appropriate.
- **Device errors**: Check that the specified devices (CPU/GPU) are available and compatible.


---

## Usage example:

```sh
3dcon --psf psf.mrc --output result.mrc --iterations 20 --lambda 1e-4 --lenA 1 --lenL 1 --lampos 1000 --epsilon 1e-5 input.mrc
```

---

## Internals

Using the notation from [High-Resolution Restoration of 3D Structures from Widefield Images with Extreme Low Signal-to-Noise-Ratio][1]

Particularly, $h(\mathbf{r})$ is the PSF, $f(\mathbf{r})$ is the input, and $\mathfrak{g}(\mathbf{r})$ is the reconstruction.

$$
\mathfrak{g} = \mathrm{argmin}_g \Big[ J(g) \Big],
$$
where
$$
J(g)=J_0(g) + J_1(g) + J_2(g)
$$

$$
$$


$$
J_0=\sum_\mathbf{r}
\left(\sum_\mathbf{r'}
h(\mathbf{r}, \mathbf{r'}) g(\mathbf{r'}) - f(\mathbf{r})\right)^2
$$

$$
J_1
=\frac{\lambda}{2} \sum_\mathbf{r}
\log\left(\epsilon
     + E^2(\mathbf{r})
\right)
\equiv\frac{\lambda}{2} \sum_\mathbf{r}
\log\left(\epsilon
     + 
     \underbrace{g^2(\mathbf{r})
     + l_A^4 \frac{\partial^2 g(\mathbf{r})}{\partial x^i \partial x^j} \frac{\partial^2 g(\mathbf{r})}{\partial x^i \partial x^j}
     + l_L^4 \frac{\partial^2 g(\mathbf{r})}{\partial x^i \partial x^i} \frac{\partial^2 g(\mathbf{r})}{\partial x^j \partial x^j}}_{E^2(\mathbf{r})}
\right)
$$

_Two quadratic scalars_ made of the second spatial derivatives possible, hence $l_A$ and  $l_L$

$$
J_2=\frac{\lambda_+}{2} \sum_\mathbf{r}
\mathtt{RelU}^2(-g(\mathbf{r}))
$$

$$
\delta J_0=
2
\sum_\mathbf{r}
\left(\sum_\mathbf{r''}
h(\mathbf{r''}, \mathbf{r})
\sum_\mathbf{r'}
\left(
h(\mathbf{r''}, \mathbf{r'}) g(\mathbf{r'})-f(\mathbf{r''})
\right)\right)\, \delta g(\mathbf{r})
$$

$$
\delta J_1=
\lambda\sum_\mathbf{r}
\left(
\frac{g(\mathbf{r}) }{\epsilon+ E^2}
+
\frac{\partial^2}{\partial x^i \partial x^j}
\left(\frac{l_A^4}{\epsilon+ E^2}
\frac{\partial^2 g(\mathbf{r})}{\partial x^i \partial x^j}
\right)
+
\frac{\partial^2}{\partial x^i \partial x^i}
\left(\frac{l_L^4}{\epsilon+ E^2}
\frac{\partial^2 g(\mathbf{r})}{\partial x^j \partial x^j}
\right)
\right)\, \delta g(\mathbf{r})
$$

$$
\delta J_2=-\lambda_+ \sum_\mathbf{r}\mathtt{RelU}(-g(\mathbf{r}))\, \delta g(\mathbf{r})
$$



---

## References
1. [High-Resolution Restoration of 3D Structures from Widefield Images with Extreme Low Signal-to-Noise-Ratio][1]


---

[1]: https://www.pnas.org/doi/10.1073/pnas.1315675110 "M.Arigovindan, J.C.Fung, D.Elnatan, V.Mennella, Y.-H.M.Chan, M.Pollard, E.Branlund, J.W.Sedat, and D.A.Agard, High-Resolution Restoration of 3D Structures from Widefield Images with Extreme Low Signal-to-Noise-Ratio"
