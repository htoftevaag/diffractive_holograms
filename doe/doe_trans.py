# --- 0) Imports
import os, io, math, json, time, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Try high-DPI plots
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('retina')
except Exception:
    pass

# Optional: GDS libraries
gds_backend = None
try:
    import gdstk
    gds_backend = 'gdstk'
except Exception:
    try:
        import gdspy
        gds_backend = 'gdspy'
    except Exception:
        gds_backend = None

print('GDS backend:', gds_backend)


# 1) Global Parameters
# Edit this cell to match your experiment and fabrication. Units are SI unless stated.

name = 'SINTEF'  # 'Julekort'

IMAGE_PATH = r'C:\Users\havardto\OneDrive - SINTEF\Bilder\sintef_logo.jpg'  # local path overrides URL if non-empty
# IMAGE_PATH = r'C:\Users\havardto\OneDrive - SINTEF\Bilder\stripes.jpg'  # local path overrides URL if non-empty
# IMAGE_PATH = r'C:\Users\havardto\OneDrive - SINTEF\Bilder\Private\SINTEF julekort.png'  # local path overrides URL if non-empty

# Image source
IMAGE_URL  = ''  # any BW image URL
TARGET_SIZE = 128  # pixels (NxN). Use power of 2 for FFT speed

# Optical design
wavelength_nm  = 532.0   # design wavelength in nm
z_image_mm     = 50.0    # image plane distance in mm
pixel_pitch_um = 10.0     # CGH pixel pitch (lattice) in micrometers
add_offaxis_tilt = True
tilt_cycles      = 0    # number of 2π phase cycles across the aperture (steers image off-axis)

# Material / phase-to-height (thin element approximation)
n_sub   = 1.46  # fused silica at ~532 nm
n_env   = 1.0   # air
h_max_nm = 3000.0  # maximum relief height (nm) allowed by process
levels   = 32      # number of discrete height levels (>=2)

# Fabrication tolerance (for Monte Carlo)
sigma_h_nm = 10.0   # 1-sigma height noise (nm)
cd_blur_um = 0.0    # pixel CD blur (um), modeled as Gaussian blur on height map
mc_samples = 8      # Monte Carlo samples

# GDS export
gds_pixel_size_um = pixel_pitch_um  # physical size per CGH pixel
gds_layer_base = 10  # base layer number; layers [base .. base+levels-1] used
gds_fname = fr'C:\Users\havardto\SINTEF_code\doe\{name}_cgh_greyscale_levels_new.gds'

# BMP export
bmp_fname = fr'C:\Users\havardto\SINTEF_code\doe\{name}_cgh_greyscale_levels_new.bmp'  # this is actually saves as a tif file

# Random seed for reproducibility
np.random.seed(2)

print(f'Configured TARGET_SIZE={TARGET_SIZE}, levels={levels}, h_max_nm={h_max_nm:.0f}')


# 2) Load / Fetch target image and pre-process
# The target image is interpreted as **amplitude** (sqrt of intensity). We normalize to [0,1].

def load_bw_image(target_size, url=None, path=None):
    if path and os.path.exists(path):
        img = Image.open(path).convert('L')
    else:
        if (url is None) or (requests is None):
            raise RuntimeError('Provide IMAGE_PATH or enable requests/URL')
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert('L')
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float64)
    # Normalize to [0,1]
    arr = (arr - arr.min()) / max(1e-9, (arr.max()-arr.min()))
    # Convert to amplitude (0..1)
    amp = np.sqrt(arr)
    return amp, img

target_amp, target_img = load_bw_image(TARGET_SIZE, IMAGE_URL, IMAGE_PATH)

# target_amp = np.ones_like(target_amp)
plt.figure(figsize=(5,5)); plt.imshow(target_amp**2, cmap='gray'); plt.title('Target Intensity'); plt.axis('on'); plt.show()


# 3) Propagation utilities (Angular Spectrum)
def angular_spectrum_propagate(U0, wavelength, dx, dz, n=1.0):
    """
    Scalar angular spectrum propagation.
    U0: complex field at z=0 (ny,nx)
    wavelength: meters
    dx: sample pitch (m)
    dz: propagation distance (m)
    n: refractive index of medium
    """
    k0 = 2*np.pi/wavelength
    ny, nx = U0.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    kx = 2*np.pi*FX
    ky = 2*np.pi*FY
    kz_sq = (n*k0)**2 - kx**2 - ky**2
    kz = np.sqrt(kz_sq + 0j)  # evanescent filtering implicit
    H = np.exp(1j * kz * dz)  # transfer function for prop over dist dz
    return np.fft.ifft2(np.fft.fft2(U0) * H)  # like C := FT(B) Wikipedia assumes propagation is just a Fourier transform (Fraunhofer approximation). 
    # My code uses angular spectrum, which is more accurate for near-field propagation.

def add_offaxis_carrier(phase, tilt_cycles):
    ny, nx = phase.shape
    x = np.linspace(0, 1, nx, endpoint=False)
    return (phase + 2*np.pi*tilt_cycles * x[None, :]) % (2*np.pi)


# 4) Gerchberg–Saxton (phase‑only)
# Weighted GS with angular‑spectrum propagation to distance *z*.

def gerchberg_saxton(target_amp, wavelength, dx, z, iters=150, amp_in=1.0, weight=0.85, offaxis_cycles=None):
    ny, nx = target_amp.shape
    # start with random phase
    U = amp_in * np.exp(1j * 2*np.pi * np.random.rand(ny, nx))
    ox = 0.0
    if offaxis_cycles:
        ox = 2*np.pi*offaxis_cycles * np.linspace(0,1,nx,endpoint=False)[None,:]
    for it in range(iters):
        U_img = angular_spectrum_propagate(U, wavelength, dx, z)
        phase_img = np.angle(U_img)
        U_tgt = (weight*target_amp + (1-weight)*np.abs(U_img)) * np.exp(1j*phase_img)  # D := Amplitude(Target) × exp(i × Phase(C))
        U = angular_spectrum_propagate(U_tgt, wavelength, dx, -z)
        U = amp_in * np.exp(1j * (np.angle(U) + ox))  # like B := Amplitude(Source) × exp(i × Phase(A))
    return (np.angle(U) + 2*np.pi) % (2*np.pi)

lam = wavelength_nm*1e-9
dx  = pixel_pitch_um*1e-6
z   = z_image_mm*1e-3

phase = gerchberg_saxton(
    target_amp, lam, dx, z,
    iters=180, amp_in=1.0, weight=0.85,
    offaxis_cycles=(tilt_cycles if add_offaxis_tilt else None)
)

plt.figure(figsize=(6,5))
plt.imshow(phase, cmap='twilight', vmin=0, vmax=2*np.pi)
# plt.imshow(phase, cmap='rainbow', vmin=0, vmax=2*np.pi)
plt.colorbar(label='phase [rad]'); plt.title('Recovered Phase at CGH Plane'); plt.axis('off'); plt.show()

print("The sample is %.2f mm" % (TARGET_SIZE * pixel_pitch_um * 10 ** -3))


# 5) Phase → Height mapping (quantized greyscale)
# Thin‑element approximation: Δφ = (2π/λ)·(n_sub − n_env)·h.
# We wrap phase to [0, 2π) then map to height ∈ [0, h_max] and quantize to `levels`.

def phase_to_height(phase, lam, n_sub, n_env, h_max_nm, levels):
    dnd = max(1e-12, (n_sub - n_env))
    # height needed for given phase
    h_full = phase * lam / (2*np.pi*dnd)  # meters 
    # fold into [0, h_max]
    h = np.mod(h_full, h_max_nm*1e-9)
    # quantize
    L = int(levels)
    bins = np.linspace(0, h_max_nm*1e-9, L, endpoint=False)
    idx  = np.digitize(h, bins, right=False)  # place in each bin
    idx  = np.clip(idx, 0, L-1)
    centers = bins + 0.5*(h_max_nm*1e-9/L)
    hq = centers[idx]
    return hq, idx

h_q, level_idx = phase_to_height(phase, lam, n_sub, n_env, h_max_nm, levels)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.imshow(h_q*1e9, cmap='viridis'); plt.colorbar(label='height [nm]'); plt.axis('off'); plt.title('Quantized Height Map')
plt.subplot(1,2,2); plt.hist((h_q*1e9).ravel(), bins=levels); plt.xlabel('height [nm]'); plt.ylabel('count'); plt.title('Height Distribution')
plt.tight_layout(); plt.show()


# print(h_q*1e9)

# 6) Forward model: Expected image at z
# Apply the (quantized) phase from quantized heights and propagate to the image plane.

# phase from quantized height
phase_q = (2*np.pi/lam) * (n_sub - n_env) * h_q
U0 = np.exp(1j*phase_q)
U_img = angular_spectrum_propagate(U0, lam, dx, z)
I_img = np.abs(U_img)**2
I_img /= I_img.max()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(target_amp**2, cmap='gray'); plt.title('Target Intensity'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(I_img, cmap='gray'); plt.title('Simulated Intensity @ z'); plt.axis('off')
plt.tight_layout(); plt.show()

mse  = np.mean((I_img - (target_amp**2))**2)
psnr = 10*np.log10(1.0/(mse+1e-12))
print('PSNR vs target (dB):', psnr)

# 7) Monte‑Carlo tolerance analysis
# Models **height noise** and optional **CD blur** (Gaussian blur) on the height map.
# We avoid SciPy; implement a light separable Gaussian blur with NumPy.

def gaussian_kernel_1d(sigma_pix, radius_factor=3.0):
    if sigma_pix <= 0:
        return np.array([1.0])
    r = max(1, int(math.ceil(radius_factor * sigma_pix)))
    x = np.arange(-r, r+1)
    k = np.exp(-0.5*(x/sigma_pix)**2)
    k /= k.sum()
    return k

def gaussian_blur_numpy(img, sigma_pix):
    if sigma_pix <= 0:
        return img
    k = gaussian_kernel_1d(sigma_pix)
    # Separable convolution with reflect padding
    def conv1d_reflect(a, k):
        r = (len(k)-1)//2
        # pad reflect
        pad = np.pad(a, ((0,0),(r,r)), mode='reflect')
        out = np.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = np.convolve(pad[i], k, mode='valid')
        return out
    tmp = conv1d_reflect(img, k)
    tmp = conv1d_reflect(tmp.T, k).T
    return tmp

def simulate_with_tolerances(h_map, lam, n_sub, n_env, dx, z, sigma_h_nm=0.0, cd_blur_um=0.0):
    h = h_map.copy()
    if cd_blur_um and cd_blur_um > 0:
        sigma_pix = (cd_blur_um*1e-6) / dx
        h = gaussian_blur_numpy(h, sigma_pix)
    if sigma_h_nm and sigma_h_nm > 0:
        h += np.random.randn(*h.shape) * (sigma_h_nm*1e-9)
    phi = (2*np.pi/lam) * (n_sub - n_env) * h
    U0 = np.exp(1j*phi)
    U_img = angular_spectrum_propagate(U0, lam, dx, z)
    I = np.abs(U_img)**2
    return I / (I.max()+1e-12)

imgs = []
for k in range(mc_samples):
    imgs.append(simulate_with_tolerances(h_q, lam, n_sub, n_env, dx, z, sigma_h_nm, cd_blur_um))
I_mean = np.mean(imgs, axis=0)
I_std  = np.std(imgs, axis=0)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(I_mean, cmap='gray'); plt.title('Monte‑Carlo mean'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(I_std,  cmap='magma'); plt.title('Monte‑Carlo std');  plt.axis('off')
plt.tight_layout(); plt.show()

# 8b) Save quantized height map as 10-bit TIFF
# We keep the step 5 quantized heights h_q (meters). This function writes a 10-bit TIFF where
# 0 -> 0 nm and 1023 -> h_max_nm (nm). The TIFF is stored in 16-bit container (uint16) with values
# limited to the 0..(2**bits-1) range. This keeps the data portable and lossless.

def save_quantized_height_tiff(h_q, h_max_nm, fname, pixels_per_cell=1, bits=10):
    """
    Save quantized height map as N-bit TIFF (stored in a 16-bit TIFF). Assumptions:
      - h_q is in meters (same as produced in step 5). We convert to nm before mapping.
      - h_max_nm is the maximum height in nm (so value h_max_nm -> max encoded value)

    Parameters
    ----------
    h_q : ndarray (ny, nx)
        Quantized height map in meters (from step 5)
    h_max_nm : float
        Maximum height in nm (maps to max encoded value)
    fname : str
        Output filename (will be coerced to .tif if bits > 8)
    pixels_per_cell : int
        Upscale output image by this factor (nearest neighbor)
    bits : int
        Number of useful bits (default 10, range 1..16). Values will be stored in a uint16 TIFF
        with the encoded range 0..(2**bits - 1).
    """
    # validation
    if bits < 1 or bits > 16:
        raise ValueError("bits must be between 1 and 16")
    max_val = (1 << bits) - 1

    # convert heights (m) -> nm
    h_nm = (h_q * 1e9).astype(np.float64)

    # Map 0..h_max_nm -> 0..max_val
    # guard division by zero
    if h_max_nm <= 0:
        raise ValueError("h_max_nm must be positive")

    out = np.round((h_nm / float(h_max_nm)) * float(max_val))
    out = np.clip(out, 0, max_val).astype(np.uint16)

    # Prepare filename extension -> TIFF for >8 bits
    base, ext = os.path.splitext(fname)
    if bits > 8 and ext.lower() not in ('.tif', '.tiff'):
        fname = base + '.tif'

    # Create PIL image from uint16 array
    img = Image.fromarray(out)  # uint16

    if pixels_per_cell and pixels_per_cell > 1:
        img = img.resize((out.shape[1] * pixels_per_cell, out.shape[0] * pixels_per_cell), resample=Image.NEAREST)

    # Save as 16-bit TIFF
    img.save(fname, format='TIFF')
    print(f"Saved quantized height TIFF: {fname} (bits={bits}, range=0..{max_val})")
    return fname

# 9) Save quantized height map to 10-bit TIFF (simple usage)
# This cell uses the save_quantized_height_tiff helper from 8b to export the quantized height map (h_q).

out_path = fr'C:\Users\havardto\SINTEF_code\doe\{name}_cgh_quantized_height_10bit.tif'
save_quantized_height_tiff(h_q, h_max_nm, out_path, pixels_per_cell=1, bits=10)
print('Done — quantized height saved to', out_path)
