"""Utility functions."""
import logging
import numpy as np
from ..event_generation.constants import Constants

logger = logging.getLogger(__name__)

def sph_to_cart_jnp(theta, phi=0):
    """Transform spherical to cartesian coordinates."""
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

    return jnp.asarray([x, y, z], dtype=jnp.float64)


def proposal_setup():
    """Set up a proposal propagator."""
    try:
        import proposal as pp
    except ImportError as e:
        logger.critical("Could not import proposal!")
        raise e

    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(500, 1, False),
    }

    cross = pp.crosssection.make_std_crosssection(
        **args
    )  # use the standard crosssections
    collection = pp.PropagationUtilityCollection()

    collection.displacement = pp.make_displacement(cross, True)
    collection.interaction = pp.make_interaction(cross, True)
    collection.time = pp.make_time(cross, args["particle_def"], True)

    utility = pp.PropagationUtility(collection=collection)

    detector = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1e20)
    density_distr = pp.density_distribution.density_homogeneous(
        args["target"].mass_density
    )
    prop = pp.Propagator(args["particle_def"], [(detector, utility, density_distr)])
    return prop


def get_zen_azi(direc):
    """Convert a cartesian direction into zenith / azimuth (IC convention)."""
    r = np.linalg.norm(direc)
    theta = 0
    if direc[2] / r <= 1:
        theta = np.arccos(direc[2] / r)
    else:
        if direc[2] < 0:
            theta = np.pi
    if theta < 0:
        theta += 2 * np.pi
    phi = 0
    if (direc[0] != 0) or (direc[1] != 0):
        phi = np.arctan2(direc[1], direc[0])
    if phi < 0:
        phi += 2 * np.pi
    zenith = np.pi - theta
    azimuth = phi + np.pi
    if zenith > np.pi:
        zenith -= 2 * np.pi - zenith
    azimuth -= int(azimuth / (2 * np.pi)) * (2 * np.pi)
    return zenith, azimuth

def is_in_cylinder(radius, height, pos):
    """Test whether a position vector is inside a cylinder."""
    return (np.sqrt(pos[0] ** 2 + pos[1] ** 2) < radius) & (np.abs(pos[2]) < height / 2)

def track_isects_cyl(radius, height, pos, direc):
    """Check if a track intersects a cylinder."""
    x = pos[0]
    y = pos[1]
    z = pos[2]

    theta, phi = get_zen_azi(direc)

    sinph = np.sin(phi)
    cosph = np.cos(phi)
    sinth = np.sin(theta)
    costh = np.cos(theta)

    b = x * cosph + y * sinph
    d = b * b + radius * radius - x * x - y * y
    h = (np.nan, np.nan)
    r = (np.nan, np.nan)

    if d > 0:
        d = np.sqrt(d)
        # down-track distance to the endcaps
        if costh != 0:
            h = sorted(((z - height / 2) / costh, (z + height / 2) / costh))
        # down-track distance to the side surfaces
        if sinth != 0:
            r = sorted(((b - d) / sinth, (b + d) / sinth))

        if costh == 0:
            if (z > -height / 2) & (z < height / 2):
                h = r
            else:
                h = (np.nan, np.nan)
        elif sinth == 0:
            if np.sqrt(x ** 2 + y ** 2) >= radius:
                h = (np.nan, np.nan)
        else:

            if (h[0] >= r[1]) or (h[1] <= r[0]):
                h = (np.nan, np.nan)
            else:
                h = max(h[0], r[0]), min(h[1], r[1])
    return h

def deposited_energy(det, record):
    """Calculate the deposited energy inside the detector outer hull."""
    dep_e = 0
    for source in record.sources:
        if is_in_cylinder(det.outer_cylinder[0], det.outer_cylinder[1], source.pos):
            dep_e += source.amp
    return dep_e

def event_labelling(track_records, strack_records, cascade_records=None, det_hull=(50.0, 1000.0), updown=True, tolerance=10):

    tolerance = tolerance*np.pi/180
    
    if cascade_records is not None:
        cascade_labels = []
        if updown:
            for r in cascade_records:
                if is_in_cylinder(det_hull[0], det_hull[1], r.mc_info[0]["pos"]):
                    cascade_labels.append(3) # contained cascade
                else:
                    cascade_labels.append(4) # uncontained cascade
                    
        else:
            for r in cascade_records:
                if is_in_cylinder(det_hull[0], det_hull[1], r.mc_info[0]["pos"]):
                    cascade_labels.append(0) # contained cascade
                else:
                    cascade_labels.append(3) # uncontained cascade
            
            
    track_labels = []
    
    if updown:
        for i,r in enumerate(track_records):
            direc = track_records[i].mc_info[0]['dir']
            pos = track_records[i].mc_info[0]['pos']
            dir_zen = get_zen_azi(direc)

            if dir_zen[0]>(0-tolerance) and dir_zen[0]<tolerance:
                track_labels.append(0) # downgoing track
            elif dir_zen[0]>(np.pi-tolerance) and dir_zen[0]<(np.pi+tolerance):
                track_labels.append(1) # upgoing track
            elif track_isects_cyl(det_hull[0], det_hull[1], pos, direc) != (np.nan, np.nan):
                track_labels.append(2) # throughgoing track
                #print('1')
            else:
                track_labels.append(3) # skimming track
    
    else:
        for r in track_records:
            if track_isects_cyl(det_hull[0], det_hull[1], r.mc_info[0]["pos"], r.mc_info[0]["pos"]) != (np.nan, np.nan):
                track_labels.append(1) # throughgoing track
            else:
                track_labels.append(3) # skimming track
                
                
    strack_labels = []
    
    if updown:
        for r in strack_records:
            if is_in_cylinder(det_hull[0], det_hull[1], r.mc_info[0]["pos"]):
                if r.mc_info[0]["energy"] > 500:
                    strack_labels.append(5) # starts in detector
                else:
                    strack_labels.append(3) # can't distingiush from cascade
                    
            else:
                isec = track_isects_cyl(det_hull[0], det_hull[1], r.mc_info[0]["pos"], r.mc_info[0]["dir"])
                if isec != (np.nan, np.nan):
                    if isec[1] < 0:
                        strack_labels.append(3) # intersection is in the opposite direction
                    elif isec[1] < r.mc_info[0]["length"]:
                        strack_labels.append(1) # throughgoing track

                    elif isec[0] > r.mc_info[0]["length"]:
                        strack_labels.append(3) # stops before entering

                    else:
                        strack_labels.append(6) # stopping track
                else:
                    strack_labels.append(3) # track doesn't intersect
                    
                    
    else:
        for r in strack_records:
            if is_in_cylinder(det_hull[0], det_hull[1], r.mc_info[0]["pos"]):
                if r.mc_info[0]["energy"] > 500:
                    strack_labels.append(2) # starts in detector
                else:
                    strack_labels.append(0) # can't distingiush from cascade
            else:
                isec = track_isects_cyl(det_hull[0], det_hull[1], r.mc_info[0]["pos"], r.mc_info[0]["pos"])
                if isec != (np.nan, np.nan):
                    if isec[1] < 0:
                        strack_labels.append(3) # intersection is in the opposite direction
                    elif isec[1] < r.mc_info[0]["length"]:
                        strack_labels.append(1) # throughgoing track

                    elif isec[0] > r.mc_info[0]["length"]:
                        strack_labels.append(3) # stops before entering

                    else:
                        strack_labels.append(4) # stopping track
                else:
                    strack_labels.append(3) # track doesn't intersect
            
            
            
    if cascade_records is None:
        return track_labels, strack_labels
    else:
        return cascade_labels, track_labels, strack_labels
    
    