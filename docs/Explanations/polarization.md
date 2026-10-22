# Polarimetric disk images

GRaTeR-JAX can return `Q`, `U`, `Qphi`, `Uphi`, and `PI` images under
single scattering with tangential linear polarization around the illuminating
star. The supplied phase function must describe **polarized scattering**.
Selecting a polarization output does not calculate a polarization fraction from
a total-intensity model.

For an existing `Optimizer` instance:

```python
q = optimizer.get_model(output="Q")
u = optimizer.get_model(output="U")
qphi = optimizer.get_model(output="Qphi")
uphi = optimizer.get_model(output="Uphi")
pi = optimizer.get_model(output="PI")
```

`get_model()` preserves the existing scalar image by default. A per-call output
selection does not change the optimizer's fitting configuration. To fit one
polarization image, supply `output="Qphi"` (or another product) when constructing
`Optimizer`, along with the existing model and parameter arguments. Its model,
likelihood, SciPy fitting, MCMC, and autodiff paths then use that product. Targets
and error maps remain two-dimensional; joint multi-channel fitting is not
provided.

## Coordinates and conventions

Angles are measured from positive x toward positive y, where x is the array
column and y is the array row. The stellar position is `x_center, y_center` in
pixel coordinates, matching the final shifted image. This is the illuminating
star's location, not an eccentric ring's geometric center. Sky orientation and
array display origin must be accounted for when comparing observations.

For `phi = atan2(y - y_star, x - x_star)` and intrinsic polarized intensity P:

- `Q = -P * cos(2 * phi)`
- `U = -P * sin(2 * phi)`
- `Qphi = -Q * cos(2 * phi) - U * sin(2 * phi)`
- `Uphi = Q * sin(2 * phi) - U * cos(2 * phi)`

Positive Q describes polarization along x, positive U along +45 degrees, and
positive Qphi tangential polarization. The exact stellar pixel has zero assigned
polarization because its azimuth is undefined. Without a PSF, Qphi equals P and
Uphi is zero away from that pixel.

## PSF and instrument response

The model converts the intrinsic, positioned disk into Q and U **before** PSF
convolution. It convolves the signed Q and U images separately, applies flux
scaling and the existing scalar throughput map, and then calculates Qphi, Uphi,
and `PI = sqrt(Q**2 + U**2)`. PI is therefore the observed polarized amplitude,
not the scalar P image independently blurred by the PSF. Opposite Stokes signals
can cancel during convolution, and observed Uphi can be nonzero.

Gaussian and empirical PSFs are supported. Additive PSF backgrounds are removed
from polarization products; the default scalar image retains its existing
behavior. Polarization outputs require `StellarPSFModel=None`, since stellar
intensity templates do not specify stellar Stokes signals. The scalar throughput
assumes equal response in Q and U, without instrumental polarization or cross-talk.
Winnie roll-dependent PSFs currently reject polarization outputs: they require
additional polarization-basis transformations across spacecraft rolls.

For intrinsic products, use an optimizer with `PSFModel=None`. Pure JAX conversion
utilities are also available in `grater_jax.disk_model.polarization`.
