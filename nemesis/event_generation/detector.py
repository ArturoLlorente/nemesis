"""Collection of classes implementing a detector."""
import itertools

import awkward as ak
import numpy as np
import scipy.stats

class Module(object):
    """
    Detection module.

    Attributes:
        pos: np.ndarray
            Module position (x, y, z)
        noise_rate: float
            Noise rate in 1/ns
        efficiency: float
            Module efficiency (0, 1]
        self.key: collection
            Module identifier
    """

    def __init__(self, pos, key, noise_rate=1, efficiency=0.2):
        """Initialize a module."""
        self.pos = pos
        self.noise_rate = noise_rate
        self.efficiency = efficiency
        self.key = key

    def __repr__(self):
        """Return string representation."""
        return repr(
            f"Module {self.key}, {self.pos} [m], {self.noise_rate} [Hz], {self.efficiency}"
        )
        
class Detector(object):
    """
    A collection of modules.

    Attributes:
        modules: List
        module_coords: np.ndarray
            N x 3 array of (x, y z) coordinates
        module_coords_ak: ak.array
            Awkward array representation of the module coordinates
        module_efficiencies: np.ndarray
            N array of the module efficiences
    """

    def __init__(self, modules):
        """Initialize detector."""
        self.modules = modules
        self.module_coords = np.vstack([m.pos for m in self.modules])
        self.module_coords_ak = ak.Array(self.module_coords)
        self.module_efficiencies = np.asarray([m.efficiency for m in self.modules])
        self.module_noise_rates = np.asarray([m.noise_rate for m in self.modules])
        
        self._outer_radius = np.linalg.norm(self.module_coords, axis=1).max()
        self._outer_cylinder = (
            np.linalg.norm(self.module_coords[:, :2], axis=1).max(),
            2 * np.abs(self.module_coords[:, 2].max()),
        )
        self._n_modules = len(modules)

    @property
    def n_modules(self):
        return self._n_modules

    @property
    def outer_radius(self):
        return self._outer_radius

    @property
    def outer_cylinder(self):
        return self._outer_cylinder
    
    
def calculate_x_displacement(module_positions_z, v_x, buoy=30, interp='cubic'):
    
    
    if buoy == 0 and interp == 'linear':
        c_ij = np.array([[0, 0, 0, 0],[0,  1.89675157e-02, -1.88782797e-05,  6.86966811e-09],[-4.12837591e-11,  3.08289997e+00, -2.57124952e-03,  7.68681299e-07],[ 9.25370891e-14, -2.64904419e+00,  2.89333720e-03, -1.13182809e-06]])    
    elif buoy == 0 and interp == 'cubic':
        c_ij = np.array([[0, 0, 0, 0],[0,  7.68639985e-03, -8.34901060e-06,  3.29026809e-09],[-4.92800042e-11,  3.24845404e+00, -2.74166773e-03,  8.32680544e-07],[ 1.10036980e-13, -3.27289333e+00,  3.54627567e-03, -1.37910532e-06]])
    elif buoy == 30 and interp == 'linear':
        c_ij = np.array([[0, 0, 0, 0],[0, -8.81316113e-03,  1.34229894e-05, -6.12599295e-09],[-4.32489363e-11,  4.10936499e+00, -1.92306436e-03,  3.42335732e-08],[ 9.77828929e-14, -2.54847112e+00,  1.85051221e-03, -4.53154300e-07]])
    elif buoy == 30 and interp == 'cubic':
        c_ij = np.array([[0, 0, 0, 0],[0, -2.12205039e-02,  2.00446217e-05, -6.77415205e-09],[-5.00985802e-11,  4.26568967e+00, -2.02259913e-03,  5.40164296e-08],[ 1.12826415e-13, -3.07903857e+00,  2.19302865e-03, -5.20736886e-07]])
    else:
        raise ValueError('Buoyancy not implemented')    
        
    A_x_all = []
    for index in range(20):
        A_x = 0
        z0 = module_positions_z[index]
        for i in range(4):
            for j in range(4):
                A_x += c_ij[i][j]*(v_x**i)*(z0**j)
        A_x_all.append(A_x)
    return A_x_all

def calculate_new_z(A_x_all, length_btw_modules = 52.63156):
    A_z_all = []
    for index in range(1, 21):
        A_z = 0
        for j in range(1,index):
            A_z += np.sqrt(length_btw_modules**2 - (A_x_all[j] - A_x_all[j-1])**2)
        A_z_all.append(A_z)    
    return A_z_all
    
def make_line(x, y, n_z, dist_z, rng, baseline_noise_rate, line_id, efficiency=0.2, v_x = 0.2, buoy_weight = 30, interp_type = 'cubic'):
    """
    Make a line of detector modules.

    The modules share the same (x, y) coordinate and are spaced along the z-direction.

    Parameters:
        x, y: float
            (x, y) position of the line
        n_z: int
            Number of modules per line
        dist_z: float
            Spacing of the detector modules in z
        rng: RandomState
        baseline_noise_rate: float
            Baseline noise rate in 1/ns. Will be multiplied to gamma(1, 0.25) distributed
            random rates per module.
        line_id: int
            Identifier for this line
        v_x: float
            current velocity in x direction (m/s)
        buoy_weight: float
            buoyancy weight (kg) [0, 30]
        interp_type: str
            interpolation type for the string bending ['linear', 'cubic']
    """
    modules = []
    
    # caclulate the z position of the modules
    module_positions_z = np.linspace(0, dist_z * n_z, n_z)
    length_btw_modules = module_positions_z[1] - module_positions_z[0]
    
    module_x_displacement = calculate_x_displacement(module_positions_z, v_x, buoy_weight, interp_type)
    module_new_z_from0 = calculate_new_z(module_x_displacement, length_btw_modules)
    module_new_z = [z_i-500 for z_i in module_new_z_from0]
    
    for i, pos_z in enumerate(module_new_z):
        pos = np.array([x+module_x_displacement[i], y, pos_z])
        noise_rate = (
            scipy.stats.gamma.rvs(1, 0.25, random_state=rng) * baseline_noise_rate
        )
        mod = Module(
            pos, key=(line_id, i), noise_rate=noise_rate, efficiency=efficiency
        )
        modules.append(mod)
        
    
    return modules

def make_triang(
    side_len,
    oms_per_line=20,
    dist_z=50,
    dark_noise_rate=16 * 1e-5,
    rng=np.random.RandomState(0),
    efficiency=0.5,
    v_x = 0,
    buoy_weight = 0,
    interp_type = 'cubic'
):

    height = np.sqrt(side_len**2 - (side_len / 2) ** 2)

    modules = make_line(
        -side_len / 2,
        -height / 3,
        oms_per_line,
        dist_z,
        rng,
        dark_noise_rate,
        0,
        efficiency=efficiency,
        v_x=v_x,
        buoy_weight=buoy_weight,
        interp_type=interp_type
    )
    modules += make_line(
        side_len / 2,
        -height / 3,
        oms_per_line,
        dist_z,
        rng,
        dark_noise_rate,
        1,
        efficiency=efficiency,
        v_x=v_x,
        buoy_weight=buoy_weight,
        interp_type=interp_type
    )
    modules += make_line(
        0,
        2 / 3 * height,
        oms_per_line,
        dist_z,
        rng,
        dark_noise_rate,
        2,
        efficiency=efficiency,
        v_x=v_x,
        buoy_weight=buoy_weight,
        interp_type=interp_type
    )

    det = Detector(modules)

    return det


def sample_cylinder_surface(height, radius, n, rng=np.random.RandomState(1337)):
    """Sample points on a cylinder surface."""
    side_area = 2 * np.pi * radius * height
    top_area = 2 * np.pi * radius ** 2

    ratio = top_area / (top_area + side_area)

    is_top = rng.uniform(0, 1, size=n) < ratio
    n_is_top = is_top.sum()
    samples = np.empty((n, 3))
    theta = rng.uniform(0, 2 * np.pi, size=n)

    # top / bottom points

    r = radius * np.sqrt(rng.uniform(0, 1, size=n_is_top))

    samples[is_top, 0] = r * np.sin(theta[is_top])
    samples[is_top, 1] = r * np.cos(theta[is_top])
    samples[is_top, 2] = rng.choice(
        [-height / 2, height / 2], replace=True, size=n_is_top
    )

    # side points

    r = radius
    samples[~is_top, 0] = r * np.sin(theta[~is_top])
    samples[~is_top, 1] = r * np.cos(theta[~is_top])
    samples[~is_top, 2] = rng.uniform(-height / 2, height / 2, size=n - n_is_top)

    return samples

def sample_cylinder_volume(height, radius, n, rng=np.random.RandomState(1337)):
    """Sample points in cylinder volume."""
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = radius * np.sqrt(rng.uniform(0, 1, size=n))
    samples = np.empty((n, 3))
    samples[:, 0] = r * np.sin(theta)
    samples[:, 1] = r * np.cos(theta)
    samples[:, 2] = rng.uniform(-height / 2, height / 2, size=n)
    return samples


def sample_direction(n_samples, rng=np.random.RandomState(1337)):
    """Sample uniform directions."""
    cos_theta = rng.uniform(-1, 1, size=n_samples)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0, 2 * np.pi)

    samples = np.empty((n_samples, 3))
    samples[:, 0] = np.sin(theta) * np.cos(phi)
    samples[:, 1] = np.sin(theta) * np.sin(phi)
    samples[:, 2] = np.cos(theta)

    return samples

def generate_noise(det, time_range, rng=np.random.RandomState(1337)):
    """Generate detector noise in a time range."""
    all_times_det = []
    dT = np.diff(time_range)
    for idom in range(len(det.modules)):
        noise_amp = rng.poisson(det.modules[idom].noise_rate * dT)
        times_det = rng.uniform(*time_range, size=noise_amp)
        all_times_det.append(times_det)

    return ak.sort(ak.Array(all_times_det))

def trigger(det, event_times, mod_thresh=8, phot_thres=5):
    """
    Check a simple multiplicity condition.

    Trigger is true when at least `mod_thresh` modules have measured more than `phot_thres` photons.

    Parameters:
        det: Detector
        event_times: ak.array
        mod_thresh: int
            Threshold for the number of modules which have detected `phot_thres` photons
        phot_thres: int
            Threshold for the number of photons per module
    """
    hits_per_module = ak.count(event_times, axis=1)
    if ak.sum((hits_per_module > phot_thres)) > mod_thresh:
        return True
    return False

