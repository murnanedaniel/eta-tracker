"""
Data preparation script for GNN tracking.

This script processes the TrackML dataset and produces graph data on disk.
"""

# System
import os
import argparse
import logging
import multiprocessing as mp
from functools import partial
import time

# Externals
import yaml
import numpy as np
from scipy import optimize
import pandas as pd
import trackml.dataset

# Locals
from datasets.param_graph import Graph, save_graphs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def select_segments(hits1, hits2, phi_slope_max, z0_max):
    """
    Construct a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """
    # Start with all possible pairs of hits
    keys = ['evtid', 'r', 'phi', 'z']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))
    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
    # Filter segments according to criteria
    good_seg_mask = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)
    return hit_pairs[['index_1', 'index_2']][good_seg_mask]

def construct_graph(hits, layer_pairs,
                    phi_slope_max, z0_max,
                    feature_names, feature_scale, T_feature_names, T_feature_scale, max_tracks=40):
    """Construct one graph (e.g. from one event)"""

    # Loop over layer pairs and construct segments
    layer_groups = hits.groupby('layer')
    segments = []
    for (layer1, layer2) in layer_pairs:
        # Find and join all hit pairs
        try:
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        # Construct the segments
        segments.append(select_segments(hits1, hits2, phi_slope_max, z0_max))
    # Combine segments from all layer pairs
    segments = pd.concat(segments)

    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    unique_pids = pd.unique(hits['particle_id'])
    
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    y_edges = np.zeros(n_edges, dtype=np.float32)
        
    I = hits['hit_id']

    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[segments.index_1].values
    seg_end = hit_idx.loc[segments.index_2].values

    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    # Fill the segment labels
    pid1 = hits.particle_id.loc[segments.index_1].values
    pid2 = hits.particle_id.loc[segments.index_2].values
    
    y_edges[:] = (pid1 == pid2)
    pids = pid1*y_edges
    
    y_params = (hits[T_feature_names].values / T_feature_scale).astype(np.float32)
    
#     print("Finished constructing graph")
    # Return a tuple of the results
    return Graph(X, Ri, Ro, y_edges, y_params, pids), I

def select_hits(hits, truth, particles, pt_min=0):
    # Barrel volume and layer ids
    vlids = [(8,2), (8,4), (8,6), (8,8),
             (13,2), (13,4), (13,6), (13,8),
             (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    # Calculate particle transverse momentum
    pt = np.sqrt(particles.px**2 + particles.py**2)
    # True particle selection.
    # Applies pt cut, removes all noise hits.
    particles = particles[pt > pt_min]
    particles = particles.join(pd.DataFrame(pt, columns=["pt"]))
    
    truth = (truth[['hit_id', 'particle_id', 'tpx', 'tpy', 'tpz']]
             .merge(particles[['particle_id', 'q', 'pt']], on='particle_id'))
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = (hits[['hit_id', 'x', 'y', 'z', 'layer']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id', 'pt', 'q', 'tpx', 'tpy', 'tpz']], on='hit_id'))
    # Remove duplicate hits
    hits = hits.loc[
        hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
    ]
    
    # Remove tracks with less than 3 hits
    pid_counts = hits.groupby(['particle_id']).size().to_frame(name = 'pidcount').reset_index()
    pid_counts = pid_counts[pid_counts.pidcount > 2]    
    
    hits = hits.loc[hits['particle_id'].isin(pid_counts['particle_id'])]
    
    #This also removes tracks with less than 3 hits, but doesn't handle duplicates
#     hits = hits[hits['nhits'] > 2]
#     print("Finished selecting hits")
    return hits

def split_detector_sections(hits, phi_edges, eta_edges):
    """Split hits according to provided phi and eta boundaries."""
    hits_sections = []
    # Loop over sections
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i+1]
        # Select hits in this phi section
        phi_hits = hits[(hits.phi > phi_min) & (hits.phi < phi_max)]
        # Center these hits on phi=0
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2
        phi_hits = phi_hits.assign(phi=centered_phi, phi_section=i)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j+1]
            # Select hits in this eta section
            eta = calc_eta(phi_hits.r, phi_hits.z)
            sec_hits = phi_hits[(eta > eta_min) & (eta < eta_max)]
            hits_sections.append(sec_hits.assign(eta_section=j))
    return hits_sections

def get_track_parameters(x, y, z):
    # find the center of helix in x-y plane
    def calc_R(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def fnc(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    r3 = np.sqrt(x**2 + y**2 + z**2)
    p_zr0 = np.polyfit(r3, z, 1, full=True)
    res0 = p_zr0[1][0]/x.shape[0]
    p_zr = p_zr0[0]

    theta = np.arccos(p_zr[0])
    # theta = np.arccos(z[0]/r3[0])
    eta = -np.log(np.tan(theta/2.))

    center_estimate = np.mean(x), np.mean(y)
    trans_center, ier = optimize.leastsq(fnc, center_estimate)
    x0, y0 = trans_center
    R = calc_R(*trans_center).mean()

    # d0, z0
    d0 = abs(np.sqrt(x0**2 + y0**2) - R)

    r = np.sqrt(x**2 + y**2)
    p_rz = np.polyfit(r, z, 1)
    pp_rz = np.poly1d(p_rz)
    z0 = pp_rz(d0)


    def quadratic_formula(a, b, c):
        if a == 0:
            return (-c/b, )
        x1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        x2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        return (x1, x2)

    # find the closest approaching point in x-y plane
    int_a = 1 + y0**2/x0**2
    int_b = -2*(x0 + y0**2/x0)
    int_c = x0**2 + y0**2 - R**2
    int_x0, int_x1 = quadratic_formula(int_a, int_b, int_c)
    x1 = int_x0 if abs(int_x0) < abs(int_x1) else int_x1
    y1 = y0*x1/x0
    phi = np.arctan2(y1, x1)

    # track travels colockwise or anti-colockwise
    # positive for colckwise
    xs = x[0] if x[0] != 0 else 1e-1
    ys = y[0] if y[0] != 0 else 1e-1
    is_14 = xs > 0
    is_above = y0 > ys/xs*x0
    sgn = 1 if is_14^is_above else -1

    # last entry is pT*(charge sign)
    return (d0, z0, phi, eta, 0.6*sgn*R/1000)

def process_event(prefix, output_dir, pt_min, n_eta_sections, n_phi_sections,
                  eta_range, phi_range, phi_slope_max, z0_max):
    # Load the data
    evtid = int(prefix[-9:])
    logging.info('Event %i, loading data' % evtid)
#     print('Event %i, loading data' % evtid)
    hits, particles, truth = trackml.dataset.load_event(
        prefix, parts=['hits', 'particles', 'truth'])

    # Apply hit selection
    logging.info('Event %i, selecting hits' % evtid)
    hits = select_hits(hits, truth, particles, pt_min=pt_min).assign(evtid=evtid)

    # Geometric features and scale
    feature_names = ['r', 'phi', 'z']
    feature_scale = np.array([1000., np.pi / n_phi_sections, 1000.])
    
    # Track param features and scale
#     T_feature_names = ['d0_T', 'z0_T', 'phi_T', 'eta_T', 'pt_T', 'pt', 'q', 'tpx', 'tpy', 'tpz']
#     T_feature_scale = np.array([100., 200., 5., 5., 20., 1., 1., 1., 1., 1.])
    T_feature_names = ['pt', 'q', 'tpx', 'tpy', 'tpz']
    T_feature_scale = np.array([1., 1., 1., 1., 1.])
    
    # Set up columns for new hit parameters
#     hits = hits.assign(**{'{}'.format(k):v for k,v in zip(T_feature_names[:5], np.zeros(len(hits)))})
    
    # Calculate track parameters
#     unique_pids = pd.unique(hits['particle_id'])
#     for pid in unique_pids:
#         hits.loc[hits.particle_id == pid, T_feature_names[:5]] = get_track_parameters(hits[hits.particle_id == pid]['x'].to_numpy(), hits[hits.particle_id == pid]['y'].to_numpy(), hits[hits.particle_id == pid]['z'].to_numpy())
    
    # Divide detector into sections
    #phi_range = (-np.pi, np.pi)
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections+1)
    hits_sections = split_detector_sections(hits, phi_edges, eta_edges)



    # Define adjacent layers
    n_det_layers = 10
    l = np.arange(n_det_layers)
    layer_pairs = np.stack([l[:-1], l[1:]], axis=1)

    # Construct the graph
    logging.info('Event %i, constructing graphs' % evtid)
    tic = time.time()
    graphs_all = [construct_graph(section_hits, layer_pairs=layer_pairs,
                              phi_slope_max=phi_slope_max, z0_max=z0_max,
                              feature_names=feature_names,
                              feature_scale=feature_scale, 
                                T_feature_names=T_feature_names,
                                T_feature_scale=T_feature_scale)
              for section_hits in hits_sections]
    graphs = [x[0] for x in graphs_all]
    IDs    = [x[1] for x in graphs_all]
    # pids   = [x[2] for x in graphs_all]
    toc = time.time()
    logging.info('Graph construction time: %f' % (toc-tic))
    # Write these graphs to the output directory
    try:
        base_prefix = os.path.basename(prefix)
        filenames = [os.path.join(output_dir, '%s_g%03i' % (base_prefix, i))
                     for i in range(len(graphs))]
        filenames_ID = [os.path.join(output_dir, '%s_g%03i_ID' % (base_prefix, i))
                     for i in range(len(graphs))]
        filenames_pid = [os.path.join(output_dir, '%s_g%03i_pid' % (base_prefix, i))
                     for i in range(len(graphs))]
    except Exception as e:
        logging.info(e)
    logging.info('Event %i, writing graphs', evtid)
    save_graphs(graphs, filenames)
    for ID, file_name in zip(IDs, filenames_ID):
        np.savez(file_name, ID=ID)
    # for pid, file_name in zip(pids, filenames_pid):
    #     np.savez(file_name, pid=pid)

def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f)
    if args.task == 0:
        logging.info('Configuration: %s' % config)

    # Find the input files
    input_dir = config['input_dir']
    all_files = os.listdir(input_dir)
    suffix = '-hits.csv'
    file_prefixes = sorted(os.path.join(input_dir, f.replace(suffix, ''))
                           for f in all_files if f.endswith(suffix))
    file_prefixes = file_prefixes[:config['n_files']]

    # Split the input files by number of tasks and select my chunk only
    file_prefixes = np.array_split(file_prefixes, args.n_tasks)[args.task]

    # Prepare output
    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Writing outputs to ' + output_dir)

    # Process input files with a worker pool
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, output_dir=output_dir,
                               phi_range=(-np.pi, np.pi), **config['selection'])
        pool.map(process_func, file_prefixes)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
