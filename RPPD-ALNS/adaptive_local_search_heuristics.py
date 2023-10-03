import networkx as nx
import numpy as np
import copy
import itertools

from adaptive_heuristics import *

###### Hub nodes local searches #################

def get_max_shaking_sol(all_shakings_sol: list, G: nx.DiGraph, hubs: dict):
    max_cost = 0
    max_sol = []
    for sol in all_shakings_sol:
        sol_cost = get_solution_cost(G, sol, hubs) 
        if sol_cost > max_cost:
            max_cost = sol_cost
            max_sol = copy.deepcopy(sol)
    return max_sol, max_cost

def get_rand_shaking_sol(all_shakings_sol: list, G: nx.DiGraph, hubs: dict):
    rand_ind = np.random.randint(0, len(all_shakings_sol))
    return all_shakings_sol[rand_ind], get_solution_cost(G, all_shakings_sol[rand_ind], hubs)

def local_search_rotate_hubs(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking=False):
    used_hubs = {key : list(set([h for arc, seg in plans[key].segments.items() for h in seg if h in hubs])) for key in plans}
    pn = [i for i in range(plans_number)]
    best_cost = cost
    best_sol = copy.deepcopy(plans)

    best_sum_cost = sum(get_separate_cost(G, best_sol, hubs))

    all_shakings_sol = []

    for p in itertools.permutations(pn):
        is_permutation_valid = True
        plans_copy = copy.deepcopy(plans)
        for key in plans_copy:
            for arc, seg in plans_copy[key].segments.items():
                plans_copy[key].segments[arc] = [seg[0]] + [seg[-1]]

        for i, key in enumerate(p):
            for h in used_hubs[i]:
                local_cost, plans_copy = insert_new_hub(copy.deepcopy(plans_copy), key, h, G, hubs)
                if not plans_copy or local_cost == float('inf'):
                    is_permutation_valid = False
                    break
            if not is_permutation_valid:
                break

        if not is_permutation_valid:
            continue

        new_cost = get_solution_cost(G, plans_copy, hubs)

        if is_shaking:
            all_shakings_sol.append(plans_copy)

        if new_cost < best_cost:
            best_cost = new_cost
            best_sol = copy.deepcopy(plans_copy)

    if is_shaking and all_shakings_sol:
        return get_max_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost


def remove_all_hubs(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking=False):
    modifyed_copy = copy.deepcopy(plans)
    seq = np.random.choice([i for i in range(plans_number)], int(plans_number/2))

    for key in modifyed_copy:
        if not key in seq:
            continue
        for arc, seg in modifyed_copy[key].segments.items():
            modifyed_copy[key].segments[arc] = [seg[0]] + [seg[-1]]

        modifyed_copy[key].nodes = list(set([v for arc, seg in modifyed_copy[key].segments.items() for v in seg]))

    mod_cost = get_solution_cost(G, modifyed_copy, hubs)

    return modifyed_copy, mod_cost

def local_search_insert_hub(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking=False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    all_shakings_sol = []
    # modifyed_copy = copy.deepcopy(plans)
    # for key in modifyed_copy:
    #     for arc, seg in modifyed_copy[key].segments.items():
    #         modifyed_copy[key].segments[arc] = [seg[0]] + [seg[-1]]

    #     modifyed_copy[key].nodes = list(set([v for arc, seg in modifyed_copy[key].segments.items() for v in seg]))

    # mod_cost = get_solution_cost(G, modifyed_copy, hubs)

    # sorted_plans = get_sorted_plans(G, best_sol, hubs)
    while True:
        plans_copy = copy.deepcopy(best_sol)
        is_continue = False
        for key in plans_copy:
            modifyed_copy = copy.deepcopy(plans_copy)
            for arc, seg in modifyed_copy[key].segments.items():
                modifyed_copy[key].segments[arc] = [seg[0]] + [seg[-1]]
            modifyed_copy[key].nodes = list(set([v for arc, seg in modifyed_copy[key].segments.items() for v in seg]))

            free_hubs = get_free_hubs(modifyed_copy, hubs)

            for h in free_hubs:
                local_cost, local_plans = insert_new_hub(copy.deepcopy(modifyed_copy), key, h, G, hubs)

                old_cost_plan = get_separate_cost(G, plans_copy, hubs)[key]
                new_cost_plan = get_separate_cost(G, local_plans, hubs)[key]

                # acceptance_criteria = (local_cost < best_cost)
                acceptance_criteria = (new_cost_plan < old_cost_plan and local_cost <= best_cost)

                # if is_shaking and local_cost <= best_cost:
                if is_shaking and local_cost < float('inf'):
                    all_shakings_sol.append(local_plans)

                if acceptance_criteria:
                    # best_cost = total_new_cost
                    # best_sol = copy.deepcopy(local_plans)
                    best_cost = local_cost
                    # modifyed_copy = copy.deepcopy(local_plans)
                    best_sol = copy.deepcopy(local_plans)
                    is_continue = True

        if not is_continue:
            break

    if have_plans_intersection(best_sol):
        print(f'The local_search_insert_hub procedure works incorrect')
        print_plans(best_sol)
        exit(0)

    # if is_shaking and all_shakings_sol:
    #     rand_ind = np.random.randint(0, len(all_shakings_sol))
    #     return all_shakings_sol[rand_ind], get_solution_cost(G, all_shakings_sol[rand_ind], hubs)

    if is_shaking and all_shakings_sol:
        return get_max_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost


def local_search_change_segment(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking=False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    all_shakings_sol = []
    while True:
        is_continue = False
        plans_local = copy.deepcopy(best_sol)
        for key in plans_local:
            for arc, seg in plans_local[key].segments.items():
                pcl = copy.deepcopy(plans_local)
                pcl[key].segments[arc] = [seg[0]] + [seg[-1]]
                pcl[key].nodes = list(set([v for arc, seg in pcl[key].segments.items() for v in seg]))
                pcl[key].segments[arc] = []
                pcl = test_insert_path(pcl, key, arc, G, hubs, seg[0], seg[-1])

                if not pcl:
                    continue

                if have_plans_intersection(pcl):
                    print(f'The local_search_change_segments works incorrect')
                    print_plans(pcl)
                    exit(0)

                # new_weight = get_solution_cost(G, pcl, hubs)
                old_weight = get_separate_cost(G, best_sol, hubs)[key]
                new_weight = get_separate_cost(G, pcl, hubs)[key]
                total_new_cost = get_solution_cost(G, pcl, hubs)

                # acceptance_criteria = (total_new_cost < best_cost)
                acceptance_criteria = (new_weight < old_weight and total_new_cost <= best_cost)

                # if is_shaking and total_new_cost <= best_cost:
                if is_shaking and total_new_cost < float('inf'):
                    all_shakings_sol.append(pcl)

                if acceptance_criteria:
                    best_cost = total_new_cost
                    best_sol = copy.deepcopy(pcl)
                    is_continue = True

        if not is_continue:
            break

    # if is_shaking and all_shakings_sol:
    #     rand_ind = np.random.randint(0, len(all_shakings_sol))
    #     return all_shakings_sol[rand_ind], get_solution_cost(G, all_shakings_sol[rand_ind], hubs)

    if is_shaking and all_shakings_sol:
        return get_max_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost

def local_search_swap_hubs(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking=False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    pn = [i for i in range(plans_number)]
    all_shakings_sol = []
    while True:
        is_continue = False
        plans_copy = copy.deepcopy(best_sol)
        old_plan_weights = get_separate_cost(G, best_sol, hubs)
        for p in itertools.combinations(pn, 2):
            p1, p2 = p
            plan1 = copy.deepcopy(plans_copy[p1])
            # plan1.nodes = list(set([v for arc, seg in plan1.segments.items() for v in seg]))

            plan2 = copy.deepcopy(plans_copy[p2])
            # plan2.nodes = list(set([v for arc, seg in plan2.segments.items() for v in seg]))
            hubs1 = [h for h in plan1.nodes if h in hubs]
            hubs2 = [h for h in plan2.nodes if h in hubs]

            zipped_hubs = [hubs1, hubs2]

            for z in itertools.product(*zipped_hubs):
                pcl = copy.deepcopy(plans_copy)
                v1, v2 = z

                generated_paths_weight_1, generated_paths_1 = construct_new_paths_swap(copy.deepcopy(pcl), copy.deepcopy(plan1), G, hubs, v1, v2)
                if not generated_paths_1 or generated_paths_weight_1 == float('inf'):
                    continue

                for key in generated_paths_1:
                    pcl[p1].segments[key] = generated_paths_1[key]
                    pcl[p1].nodes = list(set([v for arc, seg in pcl[p1].segments.items() for v in seg]))

                generated_paths_weight_2, generated_paths_2 = construct_new_paths_swap(copy.deepcopy(pcl), copy.deepcopy(plan2), G, hubs, v2, v1)
                if not generated_paths_2 or generated_paths_weight_2 == float('inf'):
                    continue

                for key in generated_paths_2:
                    pcl[p2].segments[key] = generated_paths_2[key]
                    pcl[p2].nodes = list(set([v for arc, seg in pcl[p2].segments.items() for v in seg]))

                pcl = remove_cycles(pcl)

                if have_plans_intersection(pcl):
                    print(f'The local_search_swap_vertices works incorrect')
                    print_plans(pcl)
                    exit(0)


                swap_cost = get_solution_cost(G, pcl, hubs)

                old_weight_1 = old_plan_weights[p1]
                old_weight_2 = old_plan_weights[p2]

                new_weight_1 = get_separate_cost(G, pcl, hubs)[p1]
                new_weight_2 = get_separate_cost(G, pcl, hubs)[p2]

                # acceptance_criteria = (swap_cost < best_cost)
                acceptance_criteria = (swap_cost <= best_cost and (new_weight_1 < old_weight_1 or new_weight_2 < old_weight_2))

                # print(f'{get_separate_cost(G, pcl, hubs)}')
                # ls_sol, ls_cost = local_search_insert_hub(pcl, G, clusters, hubs, source_v, dest_v, plans_number, swap_cost)

                # if (swap_cost <= best_cost) and is_shaking:
                if is_shaking:
                    all_shakings_sol.append(pcl)

                # if is_shaking:
                #     # acceptance_criteria = (swap_cost <= best_cost and ((new_weight_1 <= old_weight_1 and new_weight_2 < old_weight_2) or (new_weight_2 <= old_weight_2 and new_weight_1 < old_weight_1)))
                #     acceptance_criteria = (swap_cost <= best_cost and (new_weight_1 < old_weight_1 or new_weight_2 < old_weight_2) and not pcl in all_shakings_sol)

                if acceptance_criteria:
                    best_cost = swap_cost
                    best_sol = copy.deepcopy(pcl)
                    old_plan_weights = get_separate_cost(G, best_sol, hubs)
                    # print(f'{get_separate_cost(G, best_sol, hubs)}')
                    is_continue = True

        if not is_continue:
            break


    if is_shaking and all_shakings_sol:
        rand_ind = np.random.randint(0, len(all_shakings_sol))
        return all_shakings_sol[rand_ind], get_solution_cost(G, all_shakings_sol[rand_ind], hubs)

    # if is_shaking and all_shakings_sol:
    #     return get_max_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost

def local_search_move_hubs(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking = False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    pn = [i for i in range(plans_number)]
    all_shakings_sol = []
    while True:
        is_continue = False
        plans_copy = copy.deepcopy(best_sol)
        old_plan_weights = get_separate_cost(G, best_sol, hubs)
        for p_i in pn:
            pos_pi = [(a, seg, v) for a, seg in plans_copy[p_i].segments.items() for v in seg if v in hubs]
            if len(pos_pi) == 0:
                continue

            for rpn in pos_pi:
                relocate_pos = rpn

                for p_j in pn:
                    new_weight, new_plans = insert_hub(copy.deepcopy(plans_copy), p_i, p_j, relocate_pos, G, hubs)

                    if not new_weight or new_weight == float('inf'):
                        continue

                    new_plans = remove_cycles(new_plans)

                    for key in new_plans:
                        new_plans[key].nodes = list(set([v for seg in new_plans[key].segments.values() for v in seg]))

                    if have_plans_intersection(new_plans):
                        print(f'The local search move hubs works incorrect')
                        print_plans(new_plans)
                        exit(0)

                    new_weight = get_solution_cost(G, new_plans, hubs)

                    old_weight_1 = old_plan_weights[p_i]
                    old_weight_2 = old_plan_weights[p_j]

                    new_weight_1 = get_separate_cost(G, new_plans, hubs)[p_i]
                    new_weight_2 = get_separate_cost(G, new_plans, hubs)[p_j]

                    acceptance_criteria = (new_weight < best_cost)

                    if is_shaking:
                        all_shakings_sol.append(new_plans)
                        # acceptance_criteria = (new_weight <= best_cost and ((new_weight_1 <= old_weight_1 and new_weight_2 < old_weight_2) or (new_weight_2 <= old_weight_2 and new_weight_1 < old_weight_1)))

                    # if new_weight < best_cost:
                    if acceptance_criteria:
                        best_cost = new_weight
                        best_sol = copy.deepcopy(new_plans)
                        old_plan_weights = get_separate_cost(G, best_sol, hubs)
                        is_continue = True

        if not is_continue:
            break

    # if is_shaking and all_shakings_sol:
    #     rand_ind = np.random.randint(0, len(all_shakings_sol))
    #     return all_shakings_sol[rand_ind], get_solution_cost(G, all_shakings_sol[rand_ind], hubs)

    if is_shaking and all_shakings_sol:
        return get_max_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost

###### Cluster nodes local searches #################

def local_search_change_arc(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking=False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    start_cluster = [c for c in clusters if source_v in clusters[c]][0]
    dest_cluster = [c for c in clusters if dest_v in clusters[c]][0]
    all_shakings_sol = []
    while True:
        is_continue = False
        plans_copy = copy.deepcopy(best_sol)
        sorted_plans = get_sorted_plans(G, plans_copy, hubs)
        for key in sorted_plans:
            for arc, seg_w in sorted_plans[key]:
                max_plan_index = key
                max_arc = arc
                max_seg = seg_w[0]

                first_cluster = max_arc[0]
                second_cluster = max_arc[1]

                pcl = copy.deepcopy(plans_copy)

                start_nodes_0 = [(arc, seg[0]) for arc, seg in plans_copy[max_plan_index].segments.items() if arc[1] == first_cluster]
                dest_nodes_0 = [(arc, seg[-1]) for arc, seg in plans_copy[max_plan_index].segments.items() if arc[0] == first_cluster]
                if (max_arc, max_seg[-1]) in dest_nodes_0:
                    dest_nodes_0.remove((arc, max_seg[-1]))

                start_nodes_1 = [(arc, seg[0]) for arc, seg in plans_copy[max_plan_index].segments.items() if arc[1] == second_cluster]
                dest_nodes_1 = [(arc, seg[-1]) for arc, seg in plans_copy[max_plan_index].segments.items() if arc[0] == second_cluster]
                if (max_arc, max_seg[0]) in start_nodes_1:
                    start_nodes_1.remove((max_arc, max_seg[0]))

                free_cluster_nodes_1 = get_free_cluster_nodes(pcl, clusters, first_cluster, start_cluster, dest_cluster)
                if not max_seg[0] in free_cluster_nodes_1:
                    free_cluster_nodes_1.append(max_seg[0])

                for v1 in free_cluster_nodes_1:
                    is_v1_satisfied = True

                    # pcl = copy.deepcopy(plans_copy)
                    pcl[max_plan_index].segments[max_arc] = []
                    if max_seg[0] in pcl[max_plan_index].nodes:
                        pcl[max_plan_index].nodes.remove(max_seg[0])

                    if max_seg[-1] in pcl[max_plan_index].nodes:
                        pcl[max_plan_index].nodes.remove(max_seg[-1])

                    for arc_sn, sn_0 in start_nodes_0:
                        pcl = test_insert_path(pcl, max_plan_index, arc_sn, G, hubs, sn_0, v1)

                        if not pcl:
                            # no_improvements += 1
                            is_v1_satisfied = False
                            break

                    if not is_v1_satisfied:
                        continue

                    for arc_sn, dn_0 in dest_nodes_0:
                        pcl = test_insert_path(pcl, max_plan_index, arc_sn, G, hubs, v1, dn_0)

                        if not pcl:
                            # no_improvements += 1
                            is_v1_satisfied = False
                            break

                    if not is_v1_satisfied:
                        continue

                    free_cluster_nodes_2 = get_free_cluster_nodes(pcl, clusters, second_cluster, start_cluster, dest_cluster)
                    if not max_seg[-1] in free_cluster_nodes_2:
                        free_cluster_nodes_2.append(max_seg[-1])


                    # for v2 in clusters[second_cluster]:
                    for v2 in free_cluster_nodes_2:
                        is_v2_satisfied = True
                        pcl = test_insert_path(pcl, max_plan_index, max_arc, G, hubs, v1, v2)

                        if not pcl:
                            # no_improvements += 1
                            is_v2_satisfied = False
                            break

                        if not is_v2_satisfied:
                            continue

                        for arc_sn, sn_1 in start_nodes_1:
                            pcl = test_insert_path(pcl, max_plan_index, arc_sn, G, hubs, sn_1, v2)

                            if not pcl:
                                # no_improvements += 1
                                is_v2_satisfied = False
                                break

                        if not is_v2_satisfied:
                            continue

                        for arc_sn, dn_1 in dest_nodes_1:
                            pcl = test_insert_path(pcl, max_plan_index, arc_sn, G, hubs, v2, dn_1)

                            if not pcl:
                                # no_improvements += 1
                                is_v2_satisfied = False
                                break

                        if not is_v2_satisfied:
                            continue

                        if have_plans_intersection(pcl):
                            print(f'The local_search_change_arc works incorrect')
                            print_plans(pcl)
                            exit(0)

                        old_weight = get_separate_cost(G, best_sol, hubs)[max_plan_index]
                        new_weight = get_separate_cost(G, pcl, hubs)[max_plan_index]
                        total_new_cost = get_solution_cost(G, pcl, hubs)

                        # acceptance_criteria = (total_new_cost < best_cost)
                        acceptance_criteria = (total_new_cost <= best_cost and new_weight < old_weight)

                        if is_shaking:
                            all_shakings_sol.append(pcl)

                        # if new_weight < old_weight and best_cost >= total_new_cost:
                        if acceptance_criteria:
                            best_cost = total_new_cost
                            best_sol = copy.deepcopy(pcl)
                            is_continue = True

        if not is_continue:
            break

    # if is_shaking and all_shakings_sol:
    #     rand_ind = np.random.randint(0, len(all_shakings_sol))
    #     return all_shakings_sol[rand_ind], get_solution_cost(G, all_shakings_sol[rand_ind], hubs)

    if is_shaking and all_shakings_sol:
        return get_max_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost

def local_search_rotate_cluster_nodes(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking = False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    clusters_without_s_t = [key for key in clusters if not source_v in clusters[key] and not dest_v in clusters[key]]
    pn = [i for i in range(plans_number)]
    all_shakings_sol = []
    while True:
        is_continue = False
        plans_copy = copy.deepcopy(best_sol)
        for p in itertools.permutations(pn):
            is_permutation_valid = True
            for k in clusters_without_s_t:
                plans_copy_local = copy.deepcopy(plans_copy)
                arcs_cluster = [arc for arc, seg in plans_copy[p[0]].segments.items() if arc[0] == k or arc[1] == k]
                plans_cluster_nodes = {i : get_cluster_node(plans_copy, i, k, clusters) for i in p}


                outgoing_v = {i : {arc: seg[-1] for arc, seg in plans_copy[j].segments.items() if arc[0] == k} for j, i in enumerate(p)}
                ingoing_v = {i : {arc: seg[0] for arc, seg in plans_copy[j].segments.items() if arc[1] == k} for j, i in enumerate(p)}

                # if p == (0, 2, 1) and k == 5:
                #     print(f'{p=}')
                #     print(f'{plans_cluster_nodes=}')
                #     print(f'{outgoing_v=}')
                #     print(f'{ingoing_v=}')
                #     print_plans(plans_copy_local)
                #     print(f'{have_plans_intersection(plans_copy_local)=}')
                #     exit(0)


                for arc in arcs_cluster:
                    for i in p:
                        plans_copy_local[i].segments[arc] = []
                        plans_copy_local[i].nodes = list(set([v for a, seg in plans_copy_local[i].segments.items() for v in seg]))

                for new_number, i in enumerate(p):
                    plans_copy_local = construct_new_segments(plans_copy_local, new_number, G, hubs, plans_cluster_nodes[i], ingoing_v[i], outgoing_v[i])

                    # if p == (1, 0) and (k == 1) and new_number == 2:
                    #     print(f'{plans_cluster_nodes=}')
                    #     print(f'{outgoing_v=}')
                    #     print(f'{ingoing_v=}')
                    #     print(f'{new_number=}')
                    #     print(f'{i=}')
                    #     print_plans(plans_copy_local)
                    #     print(f'=============================')
                    #     exit(0)

                    if not plans_copy_local:
                        is_permutation_valid = False
                        break

                    # if k == 13:
                    #     print(f'{get_separate_cost(G, plans_copy_local, hubs)=}')
                    #     print_plans(plans_copy_local)
                        # exit(0)

                # print(f'{p=}, {is_permutation_valid=}, {k=}')
                # print_plans(plans_copy_local)
                # print(f'==================================')

                if have_plans_intersection(plans_copy_local):
                    print(f'{p=}')
                    print(f'{k=}')
                    print(f'{plans_cluster_nodes=}')
                    print(f'{outgoing_v=}')
                    print(f'{ingoing_v=}')
                    print_plans(plans_copy)
                    print(f'The local_search_rotate_cluster_nodes works incorrect')
                    print_plans(plans_copy_local)
                    exit(0)

                if not is_permutation_valid:
                    break

                plans_copy_local = remove_cycles(plans_copy_local)
                # print(f'{get_separate_cost(G, plans_copy_local, hubs)=}')

                for key in plans_copy_local:
                    plans_copy_local[key].nodes = list(set([v for arc, seg in plans_copy_local[key].segments.items() for v in seg]))

                if have_plans_intersection(plans_copy_local):
                    print(f'{p=}')
                    print(f'The local_search_rotate_cluster_nodes works incorrect')
                    print_plans(plans_copy_local)
                    exit(0)

                total_weight_new = get_solution_cost(G, plans_copy_local, hubs)

                acceptance_criteria = (total_weight_new < best_cost)

                if is_shaking and total_weight_new < float('inf'):
                    all_shakings_sol.append(plans_copy_local)

                if acceptance_criteria:
                    best_cost = total_weight_new
                    best_sol = copy.deepcopy(plans_copy_local)
                    is_continue = True

            if not is_permutation_valid:
                continue

            # print(f'{get_separate_cost(G, best_sol, hubs)=}')
            
        if not is_continue:
            break

    if is_shaking and all_shakings_sol:
        return get_max_shaking_sol(all_shakings_sol, G, hubs)
        # return get_rand_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost

def local_search_update_two_plans_segments(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking = False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    clusters_without_s_t = [key for key in clusters if not source_v in clusters[key] and not dest_v in clusters[key]]
    pn = [i for i in range(plans_number)]
    all_shakings_sol = []
    while True:
        is_continue = False
        plans_copy = copy.deepcopy(best_sol)
        old_plan_weights = get_separate_cost(G, best_sol, hubs)
        for k in clusters_without_s_t:
            for v in itertools.combinations(pn, 2):
                plans_copy_cost = get_solution_cost(G, plans_copy, hubs)

                plans_copy_local = copy.deepcopy(plans_copy)
                random_cluster = k

                p1 = v[0]
                p2 = v[1]
                plan1 = copy.deepcopy(plans_copy_local[p1])
                plan1.nodes = list(set([v for arc, seg in plan1.segments.items() for v in seg]))

                plan2 = copy.deepcopy(plans_copy_local[p2])
                plan2.nodes = list(set([v for arc, seg in plan2.segments.items() for v in seg]))

                v1 = list(set([seg[0] for arc, seg in plan1.segments.items() if arc[0] == random_cluster]))[0]
                v2 = list(set([seg[0] for arc, seg in plan2.segments.items() if arc[0] == random_cluster]))[0]

                arcs_cluster = [arc for arc, seg in plan1.segments.items() if arc[0] == random_cluster or arc[1] == random_cluster]

                from_v1 = {arc: seg[-1] for arc, seg in plan1.segments.items() if arc[0] == random_cluster}
                to_v1 = {arc: seg[0] for arc, seg in plan1.segments.items() if arc[1] == random_cluster}

                from_v2 = {arc : seg[-1] for arc, seg in plan2.segments.items() if arc[0] == random_cluster}
                to_v2 = {arc : seg[0] for arc, seg in plan2.segments.items() if arc[1] == random_cluster}


                for arc in arcs_cluster:
                    plan1.segments[arc] = []
                    plan2.segments[arc] = []

                # plan1.nodes = list(set([v for arc, seg in plan1.segments.items() for v in seg]))
                # plan2.nodes = list(set([v for arc, seg in plan2.segments.items() for v in seg]))

                plans_copy_local[p1] = plan1
                plans_copy_local[p2] = plan2

                plans_copy_local = construct_new_segments(plans_copy_local, p1, G, hubs, v2, to_v1, from_v1)

                if not plans_copy_local:
                    continue

                plans_copy_local = construct_new_segments(plans_copy_local, p2, G, hubs, v1, to_v2, from_v2)

                if not plans_copy_local:
                    continue

                plans_copy_local = remove_cycles(plans_copy_local)


                for key in plans_copy_local:
                    plans_copy_local[key].nodes = list(set([v for arc, seg in plans_copy_local[key].segments.items() for v in seg]))

                total_weight_new = get_solution_cost(G, plans_copy_local, hubs)

                if have_plans_intersection(plans_copy_local):
                    print(f'The local_search_update_random_cluster works incorrect')
                    print_plans(plans_copy_local)
                    exit(0)

                old_weight_1 = old_plan_weights[p1]
                old_weight_2 = old_plan_weights[p2]

                new_weight_1 = get_separate_cost(G, plans_copy_local, hubs)[p1]
                new_weight_2 = get_separate_cost(G, plans_copy_local, hubs)[p2]

                acceptance_criteria = (total_weight_new < best_cost)

                if is_shaking and total_weight_new < float('inf'):
                    all_shakings_sol.append(plans_copy_local)
                    # acceptance_criteria = (total_weight_new <= best_cost and ((new_weight_1 <= old_weight_1 and new_weight_2 < old_weight_2) or (new_weight_2 <= old_weight_2 and new_weight_1 < old_weight_1)))

                if acceptance_criteria:
                # if ((new_weight_1 <= old_weight_1 and new_weight_2 < old_weight_2) or (new_weight_2 <= old_weight_2 and new_weight_1 < old_weight_1)) and total_weight_new <= best_cost:
                    best_cost = total_weight_new
                    # sum_best_cost = sum_swap_cost
                    best_sol = copy.deepcopy(plans_copy_local)
                    old_plan_weights = get_separate_cost(G, best_sol, hubs)
                    is_continue = True

        if not is_continue:
            break

    if is_shaking and all_shakings_sol:
        return get_max_shaking_sol(all_shakings_sol, G, hubs)
        # return get_rand_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost

def local_search_change_cluster_nodes(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking = False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    clusters_without_s_t = [key for key in clusters if not source_v in clusters[key] and not dest_v in clusters[key]]
    start_cluster = [c for c in clusters if source_v in clusters[c]][0]
    dest_cluster = [c for c in clusters if dest_v in clusters[c]][0]

    pn = [i for i in range(plans_number)]
    all_shakings_sol = []
    # while True:
    is_continue = False
    plans_copy = copy.deepcopy(best_sol)
    for worse_plan_index in pn:
        for k in clusters_without_s_t:
            plans_copy_local_cluster = copy.deepcopy(plans_copy)
            # plans_copy_local_cost_cluster = get_solution_cost(G, plans_copy_local_cluster, hubs)
            # single_plan_cluster_cost = get_segments_weight(G, plans_copy_local_cluster[worse_plan_index].segments.values(), hubs)
            start_nodes = [(arc, seg[0]) for arc, seg in plans_copy_local_cluster[worse_plan_index].segments.items() if arc[1] == k]
            dest_nodes = [(arc, seg[-1]) for arc, seg in plans_copy_local_cluster[worse_plan_index].segments.items() if arc[0] == k]
            # current_plan_node = [seg[-1] for arc, seg in plans_copy_local_cluster[worse_plan_index].segments.items() if arc[1] == k][0]

            for sn in start_nodes:
                plans_copy_local_cluster[worse_plan_index].segments[sn[0]] = []
                # plans_copy_local_cluster[worse_plan_index].nodes = list(set([v for arc, seg in plans_copy_local_cluster[worse_plan_index].segments.items() for v in seg]))

            for dn in dest_nodes:
                plans_copy_local_cluster[worse_plan_index].segments[dn[0]] = []
                # plans_copy_local_cluster[worse_plan_index].nodes = list(set([v for arc, seg in plans_copy_local_cluster[worse_plan_index].segments.items() for v in seg]))

            chosen_cluster_nodes = get_free_cluster_nodes(plans_copy_local_cluster, clusters, k, start_cluster, dest_cluster)

            # arc_dn = [dn[0] for dn in dest_nodes if dn[0][0] == k][0]
            # dest = [dn[1] for dn in dest_nodes if dn[0][0] == k][0]

            for v in chosen_cluster_nodes:
                is_v_satisfied = True
                plans_copy_local_node = copy.deepcopy(plans_copy_local_cluster)
                # plans_copy_local_node = copy.deepcopy(plans_copy)
                for sn in start_nodes:
                    arc_sn = sn[0]
                    source = sn[1]
                    plans_copy_local_node = test_insert_path(plans_copy_local_node, worse_plan_index, arc_sn, G, hubs, source, v)
                    if not plans_copy_local_node:
                        is_v_satisfied = False
                        break

                if not is_v_satisfied:
                    continue

                for dn in dest_nodes:
                    arc_dn = dn[0]
                    dest = dn[1]
                    plans_copy_local_node = test_insert_path(plans_copy_local_node, worse_plan_index, arc_dn, G, hubs, v, dest)
                    if not plans_copy_local_node:
                        is_v_satisfied = False
                        break

                if not is_v_satisfied:
                    continue


                plans_copy_local_node = remove_cycles(plans_copy_local_node)

                if have_plans_intersection(plans_copy_local_node):
                    print(f'The local_search_change_cluster_nodes works incorrect')
                    print_plans(plans_copy_local_node)
                    exit(0)

                old_weight_plan = get_separate_cost(G, best_sol, hubs)[worse_plan_index]
                new_weight_plan = get_separate_cost(G, plans_copy_local_node, hubs)[worse_plan_index]

                total_weight_new = get_solution_cost(G, plans_copy_local_node, hubs)

                # acceptance_criteria = (total_weight_new < best_cost)
                acceptance_criteria = (total_weight_new <= best_cost and new_weight_plan < old_weight_plan)

                if is_shaking and total_weight_new < float('inf'):
                    all_shakings_sol.append(plans_copy_local_node)

                if acceptance_criteria:
                    best_cost = total_weight_new
                    best_sol = copy.deepcopy(plans_copy_local_node)
                    is_continue = True
        
        # if not is_continue:
        #     break

    if is_shaking and all_shakings_sol:
        # return get_max_shaking_sol(all_shakings_sol, G, hubs)
        return get_rand_shaking_sol(all_shakings_sol, G, hubs)

    return best_sol, best_cost

def local_search_swap_paths(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking = False):
    best_cost = cost
    best_sol = copy.deepcopy(plans)
    start_cluster = [arc[0] for arc, seg in plans[0].segments.items() if source_v in seg][0]
    dest_cluster = [arc[1] for arc, seg in plans[0].segments.items() if dest_v in seg][0]

    plans_arcs = [arc for arc in plans[0].segments]
    G_plans = nx.DiGraph(plans_arcs)
    order_paths = [p for p in nx.all_simple_paths(G_plans, start_cluster, dest_cluster)]
    pn = [i for i in range(plans_number)]
    all_shakings_sol = []

    while True:
        is_continue = False
        plans_current = copy.deepcopy(best_sol)
        old_plan_weights = get_separate_cost(G, best_sol, hubs)
        for p in itertools.combinations(pn, 2):
            for op in order_paths:
                plans_copy = copy.deepcopy(plans_current)
                chosen_order_path = op
                arcs_order_path = [(chosen_order_path[i], chosen_order_path[i+1]) for i in range(len(chosen_order_path)-1)]
                i0, i1 = p

                used_real_arcs_0 = get_used_real_arcs(plans_copy[i0].segments)
                G0_aux = nx.DiGraph(used_real_arcs_0)

                used_real_nodes_0 = list(set([v for arc, seg in plans_copy[i0].segments.items() for v in seg if arc in arcs_order_path and v != source_v and v != dest_v]))

                used_real_arcs_1 = get_used_real_arcs(plans_copy[i1].segments)
                used_real_nodes_1 = list(set([v for arc, seg in plans_copy[i1].segments.items() for v in seg if arc in arcs_order_path and v != source_v and v != dest_v]))

                plans_local_copy = copy.deepcopy(plans_copy)

                for arc in arcs_order_path:
                    plans_local_copy[i0].segments[arc] = plans_copy[i1].segments[arc]
                    plans_local_copy[i1].segments[arc] = plans_copy[i0].segments[arc]

                if not has_plan_other_nodes(plans_local_copy[i0].segments, used_real_nodes_0, source_v, dest_v) and not has_plan_other_nodes(plans_local_copy[i1].segments, used_real_nodes_1, source_v, dest_v):
                    # plans_local_copy = copy.deepcopy(plans_copy)

                    for arc in arcs_order_path:
                        plans_copy[i0].segments[arc] = plans_local_copy[i1].segments[arc]
                        plans_copy[i1].segments[arc] = plans_local_copy[i0].segments[arc]

                    plans_copy[i0].nodes = list(set([v for arc, seg in plans_copy[i0].segments.items() for v in seg]))
                    plans_copy[i1].nodes = list(set([v for arc, seg in plans_copy[i1].segments.items() for v in seg]))

                    if have_plans_intersection(plans_copy):
                        print(f'The local_search_swap_paths works incorrect')
                        print_plans(plans_copy)
                        exit(0)

                    new_cost = get_solution_cost(G, plans_copy, hubs)

                    old_weight_1 = old_plan_weights[i0]
                    old_weight_2 = old_plan_weights[i1]

                    new_weight_1 = get_separate_cost(G, plans_copy, hubs)[i0]
                    new_weight_2 = get_separate_cost(G, plans_copy, hubs)[i1]

                    acceptance_criteria = (new_cost < best_cost)

                    if is_shaking:
                        all_shakings_sol.append(plans_copy)
                        # acceptance_criteria = (new_cost <= best_cost and ((new_weight_1 <= old_weight_1 and new_weight_2 < old_weight_2) or (new_weight_2 <= old_weight_2 and new_weight_1 < old_weight_1)))

                    # if new_cost < best_cost:
                    if acceptance_criteria:
                        best_cost = new_cost
                        best_sol = copy.deepcopy(plans_copy)
                        old_plan_weights = get_separate_cost(G, best_sol, hubs)
                        is_continue = True
        
        if not is_continue:
            break

    # if is_shaking and all_shakings_sol:
    #     rand_ind = np.random.randint(0, len(all_shakings_sol))
    #     return all_shakings_sol[rand_ind], get_solution_cost(G, all_shakings_sol[rand_ind], hubs)

    if is_shaking and all_shakings_sol:
        return get_max_shaking_sol(all_shakings_sol, G, hubs)
    
    return best_sol, best_cost

def shaking_several_cluster_nodes(plans: dict, G: nx.DiGraph, clusters: dict, hubs: dict, source_v: int, dest_v: int, plans_number: int, cost: int, is_shaking = False):
    best_cost = cost
    best_sol = plans
    clusters_without_s_t = [key for key in clusters if not source_v in clusters[key] and not dest_v in clusters[key]]
    start_cluster = [c for c in clusters if source_v in clusters[c]][0]
    dest_cluster = [c for c in clusters if dest_v in clusters[c]][0]
    np.random.shuffle(clusters_without_s_t)

    plans_copy = copy.deepcopy(plans)

    pn = [i for i in range(plans_number)]
    np.random.shuffle(pn)

    for worse_plan_index in pn:
        for k in clusters_without_s_t:
            plans_copy_local_cluster = copy.deepcopy(plans_copy)
            
            start_nodes = [(arc, seg[0]) for arc, seg in plans_copy_local_cluster[worse_plan_index].segments.items() if arc[1] == k]
            dest_nodes = [(arc, seg[-1]) for arc, seg in plans_copy_local_cluster[worse_plan_index].segments.items() if arc[0] == k]

            for sn in start_nodes:
                plans_copy_local_cluster[worse_plan_index].segments[sn[0]] = []

            for dn in dest_nodes:
                plans_copy_local_cluster[worse_plan_index].segments[dn[0]] = []

            chosen_cluster_nodes = np.random.choice(get_free_cluster_nodes(plans_copy_local_cluster, clusters, k, start_cluster, dest_cluster), 1)

            for v in chosen_cluster_nodes:
                is_v_satisfied = True
                plans_copy_local_node = copy.deepcopy(plans_copy_local_cluster)
                for sn in start_nodes:
                    arc_sn = sn[0]
                    source = sn[1]
                    plans_copy_local_node = test_insert_path(plans_copy_local_node, worse_plan_index, arc_sn, G, hubs, source, v, 'Shaking')
                    if not plans_copy_local_node:
                        is_v_satisfied = False
                        break

                if not is_v_satisfied:
                    continue

                for dn in dest_nodes:
                    arc_dn = dn[0]
                    dest = dn[1]
                    plans_copy_local_node = test_insert_path(plans_copy_local_node, worse_plan_index, arc_dn, G, hubs, v, dest, 'Shaking')
                    if not plans_copy_local_node:
                        is_v_satisfied = False
                        break

                if not is_v_satisfied:
                    continue


                plans_copy_local_node = remove_cycles(plans_copy_local_node)

                if have_plans_intersection(plans_copy_local_node):
                    print(f'The local_search_change_cluster_nodes works incorrect')
                    print_plans(plans_copy_local_node)
                    exit(0)


                plans_copy = copy.deepcopy(plans_copy_local_node)

    return plans_copy, get_solution_cost(G, plans_copy, hubs)
