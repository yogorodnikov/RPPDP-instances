from queue import Full
from re import sub
import time
import sys
from sys import exit
import numpy as np
import networkx as nx
# import random
import math
from utilities import *
from copy import copy, deepcopy
import itertools
from typing import List, Tuple
from fromPCGLNS import create_partial_order, reduce_the_order
import random
import copy
from adaptive_local_search_heuristics import *
from adaptive_heuristics import *

from bidirection import bidirection
from start_solution import *

import multiprocessing as mp

TEST = False
T = float('inf')
P = 1e+6
removing_weights = []
insertion_weights = []
selection_weights = []

DEEPCOPY_TIME = 0
REM_TIME = 0
INS_TIME = 0
PLAN_COST_TIME = 0

NOISE = 0.25

BEST_COST = mp.Value('d', float('inf'))
BEST_SOL = mp.Manager().dict()

LS_MOVE_HUBS=0
LS_CHANGE_CLUSTER_NODES=1
LS_SWAP_HUBS=2
LS_UPDATE_TWO_PLANS_SEGMENTS=3
LS_SWAP_PATHS=4
LS_CHANGE_SEGMENT=5
LS_CHANGE_ARC=6
LS_INSERT_NEW_HUBS=7
LS_ROTATE_HUBS = 8

SH_SEVERAL_NODES=8


def shaking_move_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, reduced_order: nx.DiGraph):
    return local_search_move_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_swap_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, reduced_order: nx.DiGraph):
    return local_search_swap_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_update_two_plans_segments(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, reduced_order: nx.DiGraph):
    return local_search_update_two_plans_segments(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_swap_paths(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, reduced_order: nx.DiGraph):
    return local_search_swap_paths(plans, G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_change_cluster_nodes(plans: dict, Full_G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, reduced_order: nx.DiGraph):
    return local_search_change_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_change_segment(plans: dict, Full_G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, reduced_order: nx.DiGraph):
    return local_search_change_segment(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_change_arc(plans: dict, Full_G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, reduced_order: nx.DiGraph):
    return local_search_change_arc(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_insert_hub(plans: dict, Full_G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, reduced_order: nx.DiGraph):
    return local_search_insert_hub(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_random_start(plans: dict, Full_G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, reduced_order: nx.DiGraph):
    return get_start_solution_randomly(Full_G, clusters, hubs, reduced_order, plans_number, source_v, dest_v)

def shaking_rotate_hubs(plans: dict, Full_G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, reduced_order: nx.DiGraph):
    return local_search_rotate_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def shaking_rotate_cluster_nodes(plans: dict, Full_G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, reduced_order: nx.DiGraph):
    return local_search_rotate_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, True)

def run_heuristics(local_searches: list, shakings: list, start_solution: dict, plans_number: int, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, cost: int, start_time: int, time_limit: int, reduced_order: nx.DiGraph):
    global BEST_COST
    global BEST_SOL
    # plans_number = len(start_solution)
    # global_iteration = 10
    inner_iterations = 5
    n_ls = len(local_searches)
    n_sh = len(shakings)
    weights_local_searches = {i : 1 for i in range(n_ls)}
    weights_shakings = {i : 1 for i in range(n_sh)}

    best_solution = copy.deepcopy(start_solution)
    current_solution = copy.deepcopy(start_solution)
    temp_solution = copy.deepcopy(start_solution)

    best_solution_cost = cost
    current_solution_cost = cost
    temp_solution_cost = cost

    zeta_1 = 50
    zeta_2 = 20
    zeta_3 = 5

    theta = 50
    eps = 0.0001
    lamb = 0.1
    eta = 0.2

    is_continued = True

    # test_ls_indices = [0, 3, 1, 2]
    # i = 0

    # for phi_1 in range(global_iteration):
    while is_continued:
        scores_ls = {i : 0 for i in range(n_ls)}
        attempts_ls = {i: 0 for i in range(n_ls)}

        scores_sh = {i : 0 for i in range(n_sh)}
        attempts_sh = {i: 0 for i in range(n_sh)}

        for phi_2 in range(inner_iterations):
            choosen_ls_id = np.random.choice([i for i in range(n_ls)], p=[w/sum(weights_local_searches.values()) for i, w in weights_local_searches.items()])
            choosen_ls = local_searches[choosen_ls_id]

            choosen_sh_id = np.random.choice([i for i in range(n_sh)], p=[w/sum(weights_shakings.values()) for i, w in weights_shakings.items()])
            choosen_sh = shakings[choosen_sh_id]

            attempts_ls[choosen_ls_id] += 1
            attempts_sh[choosen_sh_id] += 1

            # print(f'shaking:')
            # print(f'{choosen_sh_id=}')
            temp_solution, temp_solution_cost = choosen_sh(copy.deepcopy(temp_solution), G, clusters, hubs, source_v, dest_v, plans_number, temp_solution_cost, reduced_order)
            # print(f'After SH {get_separate_cost(G, temp_solution, hubs)=}')
            # print(f'{temp_solution_cost}')
            # print(f'{print_plans(temp_solution)}')
            # prev_cost = temp_solution_cost
            # while True:
            # print(f'local search:')
            # print(f'{choosen_ls_id=}')
            temp_solution, temp_solution_cost = choosen_ls(copy.deepcopy(temp_solution), G, clusters, hubs, source_v, dest_v, plans_number, temp_solution_cost)
                # print(f'{temp_solution_cost}')
                # if temp_solution_cost < prev_cost:
                #     prev_cost = temp_solution_cost
                # else:
                #     break
            # print(f'{temp_solution_cost}')
            # print(f'{print_plans(temp_solution)}')
            # print(f'After LS {get_separate_cost(G, temp_solution, hubs)=}')
            # exit(0)

            if temp_solution_cost < best_solution_cost:
                # print(f'local_search:')
                # print(f'{choosen_ls_id}')

                # print(f'shakings:')
                # print(f'{choosen_sh_id}')
                best_solution = copy.deepcopy(temp_solution)
                best_solution_cost = temp_solution_cost
                current_solution = copy.deepcopy(temp_solution)
                current_solution_cost = temp_solution_cost
                # print(f'{choosen_sh_id}')
                # print(f'{choosen_ls_id}')

                with BEST_COST.get_lock():
                    BEST_COST.value = best_solution_cost
                    for r in range(plans_number): 
                        BEST_SOL[r] = copy.deepcopy(best_solution[r])

                print(f'Adaptive search find the better value {best_solution_cost=} by {choosen_ls_id=} at time: {time.time() - start_time}')
                scores_ls[choosen_ls_id] += zeta_1
                scores_sh[choosen_sh_id] += zeta_1
            elif temp_solution_cost < current_solution_cost:
                current_solution = copy.deepcopy(temp_solution)
                current_solution_cost = get_solution_cost(G, current_solution, hubs)

                scores_ls[choosen_ls_id] += zeta_2
                scores_sh[choosen_sh_id] += zeta_2
            elif theta > eps:
                rho = math.exp((best_solution_cost - temp_solution_cost) / theta)
                tau = np.random.randint(0, 1)

                if tau > rho:
                    temp_solution = copy.deepcopy(best_solution)
                    temp_solution_cost = get_solution_cost(G, temp_solution, hubs)
                else:
                    current_solution = copy.deepcopy(temp_solution)
                    current_solution_cost = get_solution_cost(G, current_solution, hubs)

                    scores_ls[choosen_ls_id] += zeta_3
                    scores_sh[choosen_sh_id] += zeta_3

                theta = theta * lamb
            else:
                temp_solution = copy.deepcopy(best_solution)
                temp_solution_cost = get_solution_cost(G, temp_solution, hubs)

            if time.time() - start_time >= time_limit:
                is_continued = False
                break

        for i in range(n_ls):
            if attempts_ls[i] > 0:
                weights_local_searches[i] = (1-eta) * weights_local_searches[i] + (eta * scores_ls[i]) / attempts_ls[i]

        for i in range(n_sh):
            if attempts_sh[i] > 0:
                weights_shakings[i] = (1-eta) * weights_shakings[i] + (eta * scores_sh[i]) / attempts_sh[i]


        if not is_continued or time.time() >= start_time + time_limit:
            break

    print(f'{weights_local_searches}, {weights_shakings}')
    return best_solution, get_solution_cost(G, best_solution, hubs)

def run_adaptive_search(adjacent_matrix: dict, clusters: dict, hubs: dict, start_cluster: int, dest_cluster: int,
                        reduced_order: nx.DiGraph, plans_number: int, random_seed: int, start_time: int, time_limit: int):
    if random_seed:
        np.random.seed(random_seed)

    # np.random.seed(28136)

    global BEST_COST
    global BEST_SOL

    source_v = clusters[start_cluster][0]
    dest_v = clusters[dest_cluster][0]

    Full_G = nx.DiGraph()
    Full_G.add_weighted_edges_from((v, u, c) for (v, u), c in adjacent_matrix.items() if c != -1)
    # Full_G.add_weighted_edges_from((v, u, c) for (v, u), c in adjacent_matrix.items())

    start_solution = {}
    start_solution[((13,17),0)] = [72,88]
    # start_solution[((3,17),0)] = [16,29,88]
    start_solution[((3,17),0)] = [16,88]
    start_solution[((0,13),0)] = [0,72]
    start_solution[((15,1),0)] = [84,6]
    start_solution[((5,1),0)] = [24,6]
    start_solution[((1,17),0)] = [6,88]
    start_solution[((8,3),0)] = [48,16]
    start_solution[((0,15),0)] = [0,84]
    start_solution[((0,5),0)] = [0,24]
    start_solution[((0,8),0)] = [0,48]

    start_solution[((13,17),1)] = [73,88]
    # start_solution[((3,17),1)] = [15,74,88]
    start_solution[((3,17),1)] = [15,88]
    start_solution[((0,13),1)] = [0,73]
    # start_solution[((15,1),1)] = [85,74,7]
    start_solution[((15,1),1)] = [85,7]
    start_solution[((5,1),1)] = [28,7]
    start_solution[((1,17),1)] = [7,88]
    start_solution[((8,3),1)] = [44,15]
    start_solution[((0,15),1)] = [0,85]
    start_solution[((0,5),1)] = [0,28]
    start_solution[((0,8),1)] = [0,44]

    start_solution[((13,17),2)] = [70,88]
    # start_solution[((3,17),2)] = [18,32,88]
    start_solution[((3,17),2)] = [18,88]
    start_solution[((0,13),2)] = [0,70]
    # start_solution[((15,1),2)] = [80,32,8]
    start_solution[((15,1),2)] = [80,8]
    start_solution[((5,1),2)] = [25,8]
    start_solution[((1,17),2)] = [8,88]
    start_solution[((8,3),2)] = [40,18]
    start_solution[((0,15),2)] = [0,80]
    # start_solution[((0,5),2)] = [0,32,25]
    start_solution[((0,5),2)] = [0,25]
    start_solution[((0,8),2)] = [0,40]

    start_solution[((13,17),3)] = [71,88]
    # start_solution[((3,17),3)] = [13,86,88]
    start_solution[((3,17),3)] = [13,88]
    start_solution[((0,13),3)] = [0,71]
    start_solution[((15,1),3)] = [78,3]
    # start_solution[((15,1),3)] = [78,31,3]
    # start_solution[((5,1),3)] = [26,29,3]
    start_solution[((5,1),3)] = [26,3]
    start_solution[((1,17),3)] = [3,88]
    start_solution[((8,3),3)] = [45,13]
    start_solution[((0,15),3)] = [0,78]
    start_solution[((0,5),3)] = [0,26]
    start_solution[((0,8),3)] = [0,45]

    start_solution[((13,17),4)] = [68,88]
    # start_solution[((3,17),4)] = [14,76,88]
    start_solution[((3,17),4)] = [14,88]
    start_solution[((0,13),4)] = [0,68]
    start_solution[((15,1),4)] = [83,4]
    start_solution[((5,1),4)] = [27,4]
    start_solution[((1,17),4)] = [4,88]
    start_solution[((8,3),4)] = [42,14]
    start_solution[((0,15),4)] = [0,83]
    start_solution[((0,5),4)] = [0,27]
    start_solution[((0,8),4)] = [0,42]

    start_solution[((13,17),5)] = [69,88]
    # start_solution[((3,17),5)] = [12,30,88]
    start_solution[((3,17),5)] = [12,88]
    start_solution[((0,13),5)] = [0,69]
    start_solution[((15,1),5)] = [81,1]
    start_solution[((5,1),5)] = [23,1]
    start_solution[((1,17),5)] = [1,88]
    start_solution[((8,3),5)] = [43,12]
    start_solution[((0,15),5)] = [0,81]
    start_solution[((0,5),5)] = [0,23]
    start_solution[((0,8),5)] = [0,43]

    start_solution, cost = get_start_solution(clusters, hubs, reduced_order, adjacent_matrix, plans_number, start_cluster, dest_cluster, with_weight = False)
    # plans, cost = get_start_solution_parallel(Full_G, clusters, hubs, reduced_order, plans_number, source_v, dest_v)
    # plans, cost = get_start_solution_randomly(Full_G, clusters, hubs, reduced_order, plans_number, source_v, dest_v)

    if not start_solution:
        plans, cost = get_start_solution_randomly(Full_G, clusters, hubs, reduced_order, plans_number, source_v, dest_v)
    else:
        plans = get_plans_from_solution(start_solution, source_v, dest_v, plans_number)
        cost = get_solution_cost(Full_G, plans, hubs)

    print(f'Start solution:')
    print(f'{cost}')
    print_plans(plans)
    print(f'{get_separate_cost(Full_G, plans, hubs)}')

    if have_plans_intersection(plans):
        print(f'The start solution returned incorrect plans')
        exit(0)

    # exit(0)

    best_solution = copy.deepcopy(plans)
    best_solution_cost = cost

    with BEST_COST.get_lock():
        BEST_COST.value = best_solution_cost
        for r in range(plans_number):
            BEST_SOL[r] = copy.deepcopy(best_solution[r])

    # local_searches = [local_search_move_hubs, local_search_change_cluster_nodes, local_search_swap_hubs, local_search_update_two_plans_segments, local_search_swap_paths, local_search_change_segment, local_search_change_arc, local_search_insert_hub]
    # local_searches = [local_search_insert_hub, local_search_rotate_hubs, local_search_change_segment, local_search_update_two_plans_segments, local_search_change_arc, local_search_change_cluster_nodes, local_search_swap_paths]
    # local_searches = [local_search_insert_hub, local_search_rotate_hubs, local_search_update_two_plans_segments]
    local_searches = [local_search_insert_hub, local_search_rotate_hubs, local_search_change_cluster_nodes, local_search_swap_paths, local_search_update_two_plans_segments]

    # local_searches = [local_search_rotate_hubs]

    # shakings = [shaking_move_hubs, shaking_change_cluster_nodes, shaking_swap_hubs, shaking_update_two_plans_segments, shaking_swap_paths, shaking_change_segment, shaking_change_arc, shaking_insert_hub, shaking_several_cluster_nodes, shaking_random_start]
    # shakings = [shaking_insert_hub, shaking_rotate_hubs, shaking_change_segment, shaking_update_two_plans_segments, shaking_change_arc, shaking_change_cluster_nodes, shaking_swap_paths, shaking_several_cluster_nodes, shaking_random_start]
    # shakings = [shaking_insert_hub, shaking_rotate_hubs, shaking_several_cluster_nodes, shaking_random_start]
    shakings = [shaking_insert_hub, shaking_rotate_hubs, shaking_change_cluster_nodes, shaking_several_cluster_nodes, shaking_swap_paths, shaking_random_start]

    # shakings = [shaking_rotate_hubs]

    best_solution, best_solution_cost = run_heuristics(local_searches, shakings, plans, plans_number, Full_G, clusters, hubs, source_v, dest_v, cost, start_time, time_limit, reduced_order)

    # test_ls_rotate_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_shaking_several_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_ls_insert_new_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_ls_move_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_ls_change_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_ls_swap_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_ls_update_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_ls_swap_paths(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_ls_change_segment(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_ls_change_arc(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_adaptive(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    # test_adaptive_2(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings)
    
    print(f'{get_separate_cost(Full_G, best_solution, hubs)}')
    return best_solution, best_solution_cost


def test_ls_rotate_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_search_rotate_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)

def test_shaking_several_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    sh_sol, sh_cost = shakings[SH_SEVERAL_NODES](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{sh_cost}')
    print_plans(sh_sol)
    print(f'{get_separate_cost(Full_G, sh_sol, hubs)}')
    exit(0)

def test_ls_insert_new_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_searches[LS_INSERT_NEW_HUBS](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)

def test_ls_move_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_searches[LS_MOVE_HUBS](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)

def test_ls_change_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_searches[LS_CHANGE_CLUSTER_NODES](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)    

def test_ls_swap_hubs(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_searches[LS_SWAP_HUBS](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)

def test_ls_update_cluster_nodes(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_search_update_two_plans_segments(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)

def test_ls_swap_paths(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_searches[LS_SWAP_PATHS](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)

def test_ls_change_segment(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_searches[LS_CHANGE_SEGMENT](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)

def test_ls_change_arc(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_sol, ls_cost = local_searches[LS_CHANGE_ARC](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_cost}')
    print_plans(ls_sol)
    print(f'{get_separate_cost(Full_G, ls_sol, hubs)}')
    exit(0)


def test_adaptive_2(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    ls_3_sol, ls_3_cost = local_searches[3](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{ls_3_cost}')
    print_plans(ls_3_sol)

def test_adaptive(plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost, local_searches, shakings):
    sh_1_sol, sh_1_cost = shakings[1](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    print(f'{sh_1_cost}')
    print_plans(sh_1_sol)

    ls_1_sol, ls_1_cost = local_searches[1](sh_1_sol, Full_G, clusters, hubs, source_v, dest_v, plans_number, sh_1_cost)
    print(f'{ls_1_cost}')
    print_plans(ls_1_sol)

    sh_0_sol, sh_0_cost = shakings[0](ls_1_sol, Full_G, clusters, hubs, source_v, dest_v, plans_number, ls_1_cost)
    print(f'{sh_0_cost}')
    print_plans(sh_0_sol)

    ls_1_sol, ls_1_cost = local_searches[1](sh_0_sol, Full_G, clusters, hubs, source_v, dest_v, plans_number, sh_0_cost)
    print(f'{ls_1_cost}')
    print_plans(ls_1_sol)

    sh_0_sol, sh_0_cost = shakings[0](ls_1_sol, Full_G, clusters, hubs, source_v, dest_v, plans_number, ls_1_cost)
    print(f'{sh_0_cost}')
    print_plans(sh_0_sol)

    ls_3_sol, ls_3_cost = local_searches[3](sh_0_sol, Full_G, clusters, hubs, source_v, dest_v, plans_number, sh_0_cost)
    print(f'{ls_3_cost}')
    print_plans(ls_3_sol)

    # ls_6_sol, ls_6_cost = local_searches[6](plans, Full_G, clusters, hubs, source_v, dest_v, plans_number, cost)
    # print(f'{ls_6_cost}')
    # print_plans(ls_6_sol)
    
#######################################

if __name__ == '__main__':
    graph_file = ''
    task_file = ''
    task_id = 0
    random_seed = None
    time_limit = 600

    for arg in sys.argv:
        if '-test' in arg:
            TEST = True
        if '=' in arg:
            parts = arg.split('=')
            if '-graph_file' in parts[0]:
                graph_file = parts[1]
            if '-task_file' in parts[0]:
                task_file = parts[1]
            if '-task_id' in parts[0]:
                task_id = int(parts[1])
            if '-iter_count' in parts[0]:
                iteration_count = int(parts[1])
            if '-iner_iter_count' in parts[0]:
                iner_iteration_count = int(parts[1])
            if '-max_no_imp' in parts[0]:
                max_no_improvment = int(parts[1])
            if '-seed' in parts[0]:
                random_seed = int(parts[1])
            if '-time_limit' in parts[0]:
                time_limit = int(parts[1])

    nodes_number, clusters_number, hubs_number, plans_number, ordering_graph, adjacent_matrix, clusters, hubs, start_cluster, dest_cluster = parseGraph(graph_file)
    
    order = create_partial_order(ordering_graph)
    reduced_order = reduce_the_order(order)

    print(f'nodes_number: {nodes_number}')
    print(f'clusters_number: {clusters_number}')
    print(f'hubs_number: {hubs_number}')
    print(f'plans_number: {plans_number}')
    # print(f'ordering_graph: {ordering_graph}')
    # print(f'adjacent_matrix: {adjacent_matrix}')
    print(f'clusters: {clusters}')
    print(f'hubs: {hubs}')
    print(f'start_cluster: {start_cluster}')
    print(f'dest_cluster: {dest_cluster}')
    print(f'time_limit: {time_limit}')
    print(f'seed: {random_seed}')

    print(f'Graph data was parsed')

    # print(f'{len(reduced_order[0])}')

    start_time = time.time()
    best_sol, best_cost = run_adaptive_search(adjacent_matrix, clusters, hubs, start_cluster, dest_cluster,
                        reduced_order, plans_number, random_seed, start_time, time_limit)

    print(f'cost: {best_cost}')
    print(f'sol: {print_plans(best_sol)}')
    suffix = '.rppdp'
    graph_base = (graph_file.split('/')[-1])[:-len(suffix)]

    solution_filename = f'adapt/{graph_base}.adapt_sol'

    print(f'Write the solution to {solution_filename}')

    with open(solution_filename, 'wt') as f:
        f.write(f'cost: {best_sol[0]}\n')
        f.write(f'plans: {best_sol[1]}\n')
    f.close()

    print(f'time: {time.time() - start_time}')



