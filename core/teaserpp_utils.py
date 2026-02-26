import numpy as np
from sklearn.neighbors import KDTree

import teaserpp_python
from timeit import default_timer as timer


def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    """
    Use kdtree to find mutually closest keypoints 

    ref_key: reference keypoints (source)
    test_key: test keypoints (target)
    ref: reference feature (source feature)
    test: test feature (target feature)
    """
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]
    print("n samples: " + str(n_samples))
    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test_features)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)

    j = np.arange(len(test_NN_idx))      # test indices 0..N_test-1
    i = np.squeeze(test_NN_idx)                      # matched ref index for each test j

    mutual_mask = (np.squeeze(ref_NN_idx[i]) == j)  # ref(i)'s best test is back to j
    ref_idx  = i[mutual_mask]
    test_idx = j[mutual_mask]

    ref_matched_keypoints = ref_keypoints[ref_idx]
    test_matched_keypoints = test_keypoints[test_idx]
    print("Number of matched keypoints: " + str(len(ref_idx)))

    # # # find mutually closest points
    # ref_match_idx = np.nonzero(
    #     np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    # )[0]
    # print("Number of matched keypoints: " + str(len(ref_match_idx)))

    # ref_matched_keypoints = ref_keypoints[ref_match_idx]
    # test_matched_keypoints = test_keypoints[ref_NN_idx[ref_match_idx]]

    return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints)


def compose_mat4_from_teaserpp_solution(solution):
    """
    Compose a 4-by-4 matrix from teaserpp solution
    """
    s = solution.scale
    rotR = solution.rotation
    t = solution.translation
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = rotR
    M = T.dot(R)

    if s == 1:
        M = T.dot(R)
    else:
        S = np.eye(4)
        S[0:3, 0:3] = np.diag([s, s, s])
        M = T.dot(R).dot(S)

    return M

def execute_teaser_global_registration(source, target, config):
    """
    Use TEASER++ to perform global registration
    """
    # Prepare TEASER++ Solver
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = config.cbar2
    solver_params.noise_bound = config.noise_bound
    solver_params.estimate_scaling = config.estimate_scaling
    if config.rotation_estimation_algorithm == 0:
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
    elif config.rotation_estimation_algorithm == 1:
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.FGR
        )
    solver_params.rotation_gnc_factor = config.rotation_gnc_factor
    solver_params.rotation_max_iterations = config.rotation_max_iterations
    solver_params.rotation_cost_threshold = config.rotation_cost_threshold
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # Solve with TEASER++
    start = timer()
    teaserpp_solver.solve(source, target)
    end = timer()
    est_solution = teaserpp_solver.getSolution()
    est_mat = compose_mat4_from_teaserpp_solution(est_solution)
    max_clique = teaserpp_solver.getTranslationInliersMap()
    print("Max clique size:", len(max_clique))
    final_inliers = teaserpp_solver.getTranslationInliers()
    return est_mat, max_clique, end - start
