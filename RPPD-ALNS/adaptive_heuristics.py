import numpy as np
import networkx as nx
import copy

class Plan():
    def __init__(self, plan_number: int, source_v: int, dest_v: int):
        self.plan_number = plan_number
        self.source_v = source_v
        self.dest_v = dest_v
        self.segments = None 
        self.nodes = None
        self.disjoin = False

    def add_segment(self, arc: tuple, path: list):
        if not self.segments:
            self.segments = {}
        self.segments[arc] = path

        if not self.nodes:
            self.nodes = []
        self.nodes.extend([node for node in path if not node in self.nodes])
        # self.nodes = list(set(self.nodes))

    def copy(self, orig):
        self.plan_number = orig.plan_number
        self.source_v = orig.source_v
        self.dest_v = orig.dest_v
        self.segments = copy.deepcopy(orig.segments)
        self.nodes = copy.deepcopy(orig.nodes)
        self.disjoin = orig.disjoin
        return self


    def remove_node(self, v: int):
        pred_nodes = {arc : (w, seg) for arc, seg in self.segments.items() for i, w in enumerate(seg[:-1]) if seg[i+1] == v}
        suc_nodes = {arc : (w, seg) for arc, seg in self.segments.items() for i, w in enumerate(seg[1:]) if seg[i] == v}
        self.nodes.remove(v)
        segments_to_remove = [arc for arc, seg in self.segments.items() if v in seg]
        for arc in segments_to_remove:
            self.segments.pop(arc)
        return pred_nodes, suc_nodes


    def __str__(self):
        msg = f'{self.plan_number}\n'
        for a, segment in self.segments.items():
            msg += f'edge={a} segment={segment}\n'
        # msg += f'{self.nodes}\n'
        # msg += 'Hubs stat:\n'
        # for v, stat in self.hubs_stat.items():
        #     msg += f'{v}: {stat}\n'
        # msg += 'Cluster nodes\n'
        # for i, v in self.clusters_stat.items():
        #     msg += f'cluster_{i}: {v}\n'

        return msg

def get_heaviest_segment(G: nx.DiGraph, plans: dict, hubs: dict):
    max_weight = 0
    max_arc = []
    max_seg = []
    for key in plans:
        for arc, seg in plans[key].segments.items():
            used_arcs = []
            used_hubs = []
            seg_weight = 0
            for i in range(len(seg)-1):
                if not (seg[i], seg[i+1]) in used_arcs:
                    used_arcs.append((seg[i], seg[i+1]))
                if seg[i] in hubs and not seg[i] in used_hubs:
                    used_hubs.append(seg[i])

            for a in used_arcs:
                seg_weight += G[a[0]][a[1]]["weight"]

            for h in used_hubs:
                seg_weight += hubs[h][0]

            if seg_weight > max_weight:
                max_weight = seg_weight
                max_arc = arc
                max_seg = seg

    return key, max_arc, max_seg, max_weight

def get_sorted_plans(G: nx.DiGraph, plans: dict, hubs: dict):
    sorted_plans = copy.deepcopy(plans)
    max_weight = 0
    max_arc = []
    max_seg = []
    for key in plans:
        for arc, seg in plans[key].segments.items():
            used_arcs = []
            used_hubs = []
            seg_weight = 0
            for i in range(len(seg)-1):
                seg_weight += G[seg[i]][seg[i+1]]["weight"]
                if seg[i] in hubs and not seg[i] in used_hubs:
                    used_hubs.append(seg[i])

            for h in used_hubs:
                seg_weight += hubs[h][0]

            sorted_plans[key].segments[arc] = (seg, seg_weight)
    sorted_plans = {pn: sorted(sorted_plans[pn].segments.items(), key=lambda x:x[1][1], reverse=True) for pn in sorted_plans}
    return sorted_plans


def get_segments_weight(G: nx.DiGraph, segments: list, hubs: dict):
    total_weight = 0
    # used_arcs = []
    used_hubs = []
    for seg in segments:
        for i in range(len(seg)-1):
            # if not (seg[i], seg[i+1]) in used_arcs:
                # used_arcs.append((seg[i], seg[i+1]))
            if (seg[i], seg[i+1]) in G.edges:
                total_weight += G[seg[i]][seg[i+1]]["weight"]
            else:
                return float('inf')
            if seg[i] in hubs and not seg[i] in used_hubs:
                used_hubs.append(seg[i])

    # print(f'{used_arcs}')
    # print(f'{used_hubs}')
    # for arc in used_arcs:
    #     total_weight += G[arc[0]][arc[1]]["weight"]

    for h in used_hubs:
        total_weight += hubs[h][0]
    return total_weight

def get_solution_cost(G: nx.DiGraph, plans: dict, hubs: dict):
    return max([get_segments_weight(G, plans[key].segments.values(), hubs) for key in plans])

def get_separate_cost(G: nx.DiGraph, plans: dict, hubs: dict):
    return [get_segments_weight(G, plans[key].segments.values(), hubs) for key in plans]

def is_same_cluster(clusters: dict, v1: int, v2: int):
    for key in clusters:
        if v1 in clusters[key] and v2 in clusters[key]:
            return True
    return False

def get_cluster_node(plans: dict, plan_number: int, cluster_number: int, clusters: dict):
    for arc, seg in plans[plan_number].segments.items():
        if arc[0] == cluster_number:
            return seg[0]

def is_vertex_in_cluster(clusters: dict, v: int):
    for key in clusters:
        if v in clusters[key]:
            return True
    return False

def get_free_hubs(plans: dict, hubs: dict):
    if not plans:
        return copy.deepcopy(hubs)

    free_hubs = hubs.copy()
    for p in plans:
        if not plans[p].nodes:
            continue
        for h in hubs:
            if h in plans[p].nodes and h in free_hubs:
                free_hubs.pop(h)
    return free_hubs

def get_free_cluster_nodes(plans: dict, clusters: dict, c: int, s_v: int, t_v: int):
    free_cluster_nodes = copy.deepcopy(clusters[c])
    if not plans or c == s_v or c == t_v:
        return copy.deepcopy(free_cluster_nodes)

    for key in plans:
        if not plans[key].segments:
            continue

        for arc, seg in plans[key].segments.items():
            for v in seg:
                if v in free_cluster_nodes:
                    free_cluster_nodes.remove(v)
        # if key == 3:
        #     print(f'{free_cluster_nodes}')
    return free_cluster_nodes

def count_hub_usage(plan: Plan, h: int):
    usage_h = 0
    for seg in plan.segments:
        if h in plan.segments[seg]:
            usage_h += 1
    return usage_h

def print_plans(plans: dict):
    for key, value in plans.items():
        print(f'{key}')
        print(f'{plans[key].segments}')
        print(f'{plans[key].nodes}')
    print(f'\n')

def extend_auxiliary_graph_by_free_hubs(G_full: nx.DiGraph, G_aux: nx.DiGraph, free_hubs: dict, hubs: dict):
    G1_aux = nx.DiGraph(G_aux)
    for h in free_hubs:
        ingoing_edges = G_full.in_edges(h)
        for e in ingoing_edges:
            if e[0] in G_aux.nodes:
                G1_aux.add_edge(e[0], f'{h}_1', weight=G_full[e[0]][e[1]]["weight"])
            if e[0] in free_hubs:
                G1_aux.add_edge(f'{e[0]}_2', f'{h}_1', weight=G_full[e[0]][e[1]]["weight"])

        outgoing_edges = G_full.out_edges(h)
        for e in outgoing_edges:
            if e[1] in G_aux.nodes:
                G1_aux.add_edge(f'{h}_2', e[1], weight=G_full[e[0]][e[1]]["weight"])
            if e[1] in free_hubs:
                G1_aux.add_edge(f'{h}_2', f'{e[1]}_1', weight=G_full[e[0]][e[1]]["weight"])
        G1_aux.add_edge(f'{h}_1', f'{h}_2', weight=free_hubs[h][0])

    return G1_aux

def remove_cycles(plans: dict):
    plans_copy = copy.deepcopy(plans)
    for key in plans:
        for arc, seg in plans[key].segments.items():
            used_nodes = []
            for i in range(len(seg)):
                if seg[i] in used_nodes:
                    used_nodes = used_nodes[:used_nodes.index(seg[i])+1]
                else:
                    used_nodes.append(seg[i])
            plans_copy[key].segments[arc] = used_nodes
        plans_copy[key].nodes = list(set([v for arc, seg in plans_copy[key].segments.items() for v in seg]))
    return plans_copy


def test_insert_swap_node(plans: dict, target_plan: Plan, G: nx.DiGraph, hubs: dict, v: int, from_v: int, to_v: int, seg: list, arc: tuple, generated_paths: dict):
    hubs_generated_path = [v for arc, seg in generated_paths.items() for v in seg if v in hubs]
    free_hubs = get_free_hubs(plans, hubs)
    # print(f'{free_hubs}')
    for h in hubs_generated_path:
        if h in free_hubs:
            free_hubs.pop(h)
    # hubs_generated_path_new = [v for v in hubs_generated_path if hubs_generated_path.count(v) < hubs[v][1]]
    subgraph_nodes = [v for v in target_plan.nodes if v in hubs and count_hub_usage(target_plan, v) + hubs_generated_path.count(v) < hubs[v][1]]

    if not from_v in subgraph_nodes:
        subgraph_nodes.append(from_v)
    if not v in subgraph_nodes:
        subgraph_nodes.append(v)
    if not to_v in subgraph_nodes:
        subgraph_nodes.append(to_v)

    # G_aux = nx.subgraph(G, list(set(subgraph_nodes + [from_v, v, to_v] + hubs_generated_path_new)))
    G_aux = nx.subgraph(G, list(set(subgraph_nodes)))
    G_aux = nx.DiGraph(G_aux)
    G_aux = extend_auxiliary_graph_by_free_hubs(G, G_aux, free_hubs, hubs)
    if nx.has_path(G_aux, source=from_v, target=to_v):
        inserted_part = remove_cost_edges(nx.shortest_path(G_aux, source=from_v, target=to_v, weight='weight'))
        if v == to_v:
            path_part = seg[:seg.index(from_v)] + inserted_part
        else:
            path_part = inserted_part + seg[seg.index(to_v)+1:]

        # print(f'{path_part}')

        if not arc in generated_paths:
            generated_paths[arc] = path_part
        else:
            generated_paths[arc].extend(path_part[1:])
    else:
        return None
    return generated_paths


def construct_new_paths_swap(plans: dict, target_plan: Plan, G: nx.DiGraph, hubs: dict, removed_node: int, inserted_node: int):
    prev_nodes, suc_nodes = target_plan.remove_node(removed_node)
    target_plan.nodes = list(set([v for arc, seg in target_plan.segments.items() for v in seg]))
    generated_paths = {}
    for arc, data in prev_nodes.items():
        generated_paths = test_insert_swap_node(plans, target_plan, G, hubs, inserted_node, data[0], inserted_node, data[1], arc, generated_paths)
        if not generated_paths:
            return None, None
    
    for arc, data in suc_nodes.items():
        generated_paths = test_insert_swap_node(plans, target_plan, G, hubs, inserted_node, inserted_node, data[0], data[1], arc, generated_paths)
        if not generated_paths:
            return None, None
    
    generated_paths_weight = get_segments_weight(G, generated_paths.values(), hubs)
    return generated_paths_weight, generated_paths

def remove_cost_edges(seg: list):
    updated_seg = copy.deepcopy(seg)
    for i in range(len(seg)-1):
        if '_1' in str(seg[i]) and '_2' in str(seg[i+1]):
            h = int(seg[i].split('_')[0])
            updated_seg.insert(updated_seg.index(seg[i]), h)
            updated_seg.remove(seg[i])
            updated_seg.remove(seg[i+1])
    return updated_seg


def construct_segment(plans: dict, target_plan: Plan, G: nx.DiGraph, clusters: dict, hubs: dict, routed_arc: tuple, start_node: int, dest_cluster: int, s_v: int, t_v: int):
    free_hubs = get_free_hubs(plans, hubs)
    free_cluster_nodes = get_free_cluster_nodes(plans, clusters, routed_arc[1], s_v, t_v)
    # print(f'{routed_arc[1]}, {free_cluster_nodes}')

    min_seg = []
    min_seg_weight = float('inf')

    if routed_arc[1] == dest_cluster and not clusters[routed_arc[1]][0] in free_cluster_nodes:
        free_cluster_nodes.append(clusters[routed_arc[1]][0])

    if target_plan.nodes:
        for v in target_plan.nodes:
            if v in clusters[routed_arc[1]]:
                free_cluster_nodes = [v]
                break

    for v in free_cluster_nodes:
        G_Aux = nx.DiGraph()
        subgraph_nodes = []
        if target_plan.nodes:
            subgraph_nodes = [v for v in target_plan.nodes if v in hubs and count_hub_usage(target_plan, v) < hubs[v][1]]
        subgraph_nodes.append(start_node)
        subgraph_nodes.append(v)


        G_aux = nx.subgraph(G, list(set(subgraph_nodes)))
        G_aux = nx.DiGraph(G_aux)
        G_aux = extend_auxiliary_graph_by_free_hubs(G, G_aux, free_hubs, hubs)

        if not nx.has_path(G_aux, source=start_node, target=v):
            print(f'There is not path between {start_node=}, and {v=}')
            print(f'{subgraph_nodes=}')
            # print(f'{G.out_edges(134)=}')
            # print(f'{G_aux.edges[325,"134_1"]=}, {G_aux.edges["134_1", "134_2"]=}, {G_aux.edges["134_2", 414]=}')
            print(f'{free_hubs=}')
            print(f'{free_cluster_nodes=}')
            exit(0)
            return None, None

        seg_v = remove_cost_edges(nx.shortest_path(G_aux, source=start_node, target=v, weight='weight'))
        # seg_v = remove_cost_edges(nx.shortest_path(G_aux, source=start_node, target=v))
        seg_v_weight = get_segments_weight(G, [seg_v], hubs)
        if seg_v_weight < min_seg_weight:
            min_seg_weight = seg_v_weight
            min_seg = copy.deepcopy(seg_v)

    if not min_seg_weight < float('inf'):
        print(f'There is no path for {routed_arc}, {free_cluster_nodes}, {free_hubs}, {clusters[routed_arc[0]]}, {plans[2].segments}')
        exit(0)

    return min_seg_weight, min_seg



def test_insert_hub(plans: dict, target_plan: Plan, h: int, G: nx.DiGraph, hubs: dict, dest_arc: tuple, dest_seg: list, start_h: int, dest_h: int):
    free_hubs = get_free_hubs(plans, hubs)
    subgraph_nodes = copy.deepcopy([v for v in target_plan.nodes if v in hubs and count_hub_usage(target_plan, v) < hubs[v][1]])
    if not start_h in subgraph_nodes:
        subgraph_nodes.append(start_h)
    if not dest_h in subgraph_nodes:
        subgraph_nodes.append(dest_h)
    if not h in subgraph_nodes:
        subgraph_nodes.append(h)
    # G_aux = nx.subgraph(G, subgraph_nodes + list(free_hubs.keys()))
    G_aux = nx.subgraph(G, subgraph_nodes)
    G_aux = extend_auxiliary_graph_by_free_hubs(G, G_aux, free_hubs, hubs)

    if nx.has_path(G_aux, source=start_h, target=h):
        first_part = nx.shortest_path(G_aux, source=start_h, target=h, weight='weight')
        first_part = remove_cost_edges(first_part)
        hubs_first_part = [v for v in first_part if v in hubs]
        for v in hubs_first_part:
            if count_hub_usage(target_plan, v) == hubs[v][1]-1 and v != h and v in subgraph_nodes:
                # if not v in subgraph_nodes:
                #     print(f'The {v} is not in {subgraph_nodes}')
                #     print(f'{first_part}')
                #     exit(0)
                subgraph_nodes.remove(v)
            if v in free_hubs and hubs[v][1] == 1 and v != h:
                free_hubs.pop(v)

        # G_aux = nx.subgraph(G, subgraph_nodes + list(free_hubs.keys()))
        G_aux = nx.subgraph(G, subgraph_nodes + [h, dest_h])
        G_aux = extend_auxiliary_graph_by_free_hubs(G, G_aux, free_hubs, hubs)

        if nx.has_path(G_aux, source=h, target=dest_h):
            second_part = nx.shortest_path(G_aux, source=h, target=dest_h, weight='weight')
            second_part = remove_cost_edges(second_part)
            new_segment = dest_seg[:dest_seg.index(start_h)] + first_part[:-1] + second_part[:-1] + dest_seg[dest_seg.index(dest_h):]
            target_plan.segments[dest_arc] = new_segment
            new_weight = get_segments_weight(G, target_plan.segments.values(), hubs)
            return new_weight, target_plan
        return None, None
    return None, None

def get_connection_removed_hub(plans: dict, p_from: int, p_to: int, pos: tuple, G: nx.DiGraph, hubs: dict):
    plans_copy = copy.deepcopy(plans)
    source_plan = plans_copy[p_from]

    a_source, seg_source, h_source = pos[0], pos[1], pos[2]

    pred_h = seg_source[seg_source.index(h_source)-1]
    suc_h = seg_source[seg_source.index(h_source)+1]

    pairs_to_be_connected = {a_source: (seg_source, (pred_h, suc_h))}

    if p_from != p_to:
        for arc, seg in source_plan.segments.items():
            if h_source in seg:
                pairs_to_be_connected[arc] = (seg, (seg[seg.index(h_source)-1], seg[seg.index(h_source)+1]))

    for arc, (seg, cp) in pairs_to_be_connected.items():
        free_hubs = get_free_hubs(plans_copy, hubs)
        subgraph_nodes = copy.deepcopy([v for v in source_plan.nodes if v in hubs and count_hub_usage(source_plan, v) < hubs[v][1] and v != h_source])
        subgraph_nodes.extend([cp[0], cp[1]])
        subgraph_nodes = list(set(subgraph_nodes))

        # G_aux = nx.subgraph(G, subgraph_nodes + list(free_hubs.keys()))
        G_aux = nx.subgraph(G, subgraph_nodes)
        G_aux = extend_auxiliary_graph_by_free_hubs(G, G_aux, free_hubs, hubs)

        if nx.has_path(G_aux, source=cp[0], target=cp[1]):
            new_seg_h = nx.shortest_path(G_aux, source=cp[0], target=cp[1], weight='weight')
            new_seg_h = remove_cost_edges(new_seg_h)
            start_new_seg_h = seg.index(new_seg_h[0])
            dest_new_seg_h = seg.index(new_seg_h[-1])
            plans_copy[p_from].segments[arc] = seg[:start_new_seg_h] + new_seg_h[:-1] + seg[dest_new_seg_h:]
            plans_copy = remove_cycles(plans_copy)
        else:
            return None
    return plans_copy

def insert_new_hub(plans: dict, p_to: int, h_source: int, G: nx.DiGraph, hubs: dict):
    plans_copy = copy.deepcopy(plans)
    target_plan = copy.deepcopy(plans_copy[p_to])

    h_capacity = hubs[h_source][1]

    h_i = 0
    while h_i < h_capacity:
        old_weight_target_plan = get_segments_weight(G, plans_copy[p_to].segments.values(), hubs)
        cur_max = old_weight_target_plan
        cur_plan = copy.deepcopy(plans_copy[p_to])
        for arc, seg in plans_copy[p_to].segments.items():
            for i in range(1,len(seg)):
                target_plan_copy = copy.deepcopy(plans_copy[p_to])
                target_plan_copy.segments[arc] = seg[0:i] + [h_source] + seg[i:]
                # print(f'{target_plan_copy.segments[arc]}')
                new_weight_target_plan = get_segments_weight(G, target_plan_copy.segments.values(), hubs)

                # if new_weight_target_plan and new_weight_target_plan < old_weight_target_plan:
                if new_weight_target_plan and new_weight_target_plan < cur_max:
                    # plans_copy[p_to] = target_plan_copy
                    # plans_copy = remove_cycles(plans_copy)
                    cur_max = new_weight_target_plan
                    cur_plan = copy.deepcopy(target_plan_copy)

        if cur_max < old_weight_target_plan:
            plans_copy[p_to] = copy.deepcopy(cur_plan)
            plans_copy = remove_cycles(plans_copy)
            h_i += 1
        else:
            break

    total_weight_new = max([get_segments_weight(G, plans_copy[key].segments.values(), hubs) for key in plans_copy])
    # total_weight_old = max([get_segments_weight(G, plans[key].segments.values(), hubs) for key in plans])

    # if total_weight_new < total_weight_old:
    #     return total_weight_new, plans_copy
    # else:
    #     return total_weight_old, plans

    # if p_to != 0:
    #     print(f'AFTER, {h_source}, {p_to}')
    #     print(f'{get_separate_cost(G, plans_copy, hubs)}')

    # print(f'{[h for h in plans_copy[p_to].nodes if h in hubs]}')

    return total_weight_new, plans_copy

def insert_hub(plans: dict, p_from: int, p_to: int, pos: tuple, G: nx.DiGraph, hubs: dict):
    old_weight_source_plan = get_segments_weight(G, plans[p_from].segments.values(), hubs)
    old_weight_target_plan = get_segments_weight(G, plans[p_to].segments.values(), hubs)
    plans_copy = get_connection_removed_hub(plans, p_from, p_to, pos, G, hubs)

    if not plans_copy:
        return None, None

    target_plan = plans_copy[p_to]

    a_source, seg_source, h_source = pos[0], pos[1], pos[2]
    for arc, seg in target_plan.segments.items():
        if arc == a_source and p_from == p_to:
            continue

        for i in range(len(seg)-1):
            if h_source in seg:
                continue
            new_weight_target_plan, modifyed_plan = test_insert_hub(copy.deepcopy(plans_copy), copy.deepcopy(target_plan), h_source, G, hubs, arc, seg, seg[i], seg[i+1])
            if new_weight_target_plan and new_weight_target_plan < old_weight_target_plan:
                plans_copy[p_to] = modifyed_plan
                plans_copy = remove_cycles(plans_copy)

    total_weight_new = max([get_segments_weight(G, plans_copy[key].segments.values(), hubs) for key in plans_copy])
    total_weight_old = max([get_segments_weight(G, plans[key].segments.values(), hubs) for key in plans])

    if total_weight_new < total_weight_old:
        return total_weight_new, plans_copy
    else:
        return total_weight_old, plans

def have_plans_intersection(plans: dict):
    for key1 in plans:
        for key2 in plans:
            if key1 != key2:
                if len(set(plans[key1].nodes).intersection(set(plans[key2].nodes))) != 2:
                    print(f'{set(plans[key1].nodes).intersection(set(plans[key2].nodes))}')
                    return True

    return False

def get_plans_from_solution(start_solution: dict, source_v: int, dest_v: int, plans_number: int):
    plans = {i : Plan(i, source_v, dest_v) for i in range(plans_number)}

    for key in start_solution:
        plans[key[1]].add_segment(key[0], start_solution[key])

    return plans

def construct_new_segments(plans: dict, target_plan_index: int, G: nx.DiGraph, hubs: dict, new_node: int, incoming_edges: list, outcoming_edges: list):
    for arc, ie in incoming_edges.items():
        free_hubs = get_free_hubs(plans, hubs)
        subgraph_nodes = copy.deepcopy([v for v in plans[target_plan_index].nodes if v in hubs and count_hub_usage(plans[target_plan_index], v) < hubs[v][1]])        
        # G_aux = nx.subgraph(G, subgraph_nodes + [ie, new_node] + list(free_hubs.keys()))
        if not ie in hubs:
            subgraph_nodes.append(ie)
        if not new_node in hubs:
            subgraph_nodes.append(new_node)

        G_aux = nx.subgraph(G, subgraph_nodes + [ie, new_node])
        G_aux = extend_auxiliary_graph_by_free_hubs(G, G_aux, free_hubs, hubs)
        if nx.has_path(G_aux, source=ie, target=new_node):
            new_segment = nx.shortest_path(G_aux, source=ie, target=new_node, weight='weight')
            new_segment = remove_cost_edges(new_segment)
            plans[target_plan_index].segments[arc] = new_segment
            plans[target_plan_index].nodes = list(set([v for arc, seg in plans[target_plan_index].segments.items() for v in seg]))
        else:
            return None

    for arc, oe in outcoming_edges.items():
        free_hubs = get_free_hubs(plans, hubs)
        subgraph_nodes = copy.deepcopy([v for v in plans[target_plan_index].nodes if v in hubs and count_hub_usage(plans[target_plan_index], v) < hubs[v][1]])
        # G_aux = nx.subgraph(G, subgraph_nodes + [oe, new_node] + list(free_hubs.keys()))
        G_aux = nx.subgraph(G, subgraph_nodes + [oe, new_node])
        if not oe in hubs:
            subgraph_nodes.append(oe)
        if not new_node in hubs:
            subgraph_nodes.append(new_node)
        G_aux = extend_auxiliary_graph_by_free_hubs(G, G_aux, free_hubs, hubs)
        if nx.has_path(G_aux, source=new_node, target=oe):
            new_segment = nx.shortest_path(G_aux, source=new_node, target=oe, weight='weight')
            new_segment = remove_cost_edges(new_segment)
            plans[target_plan_index].segments[arc] = new_segment
            plans[target_plan_index].nodes = list(set([v for arc, seg in plans[target_plan_index].segments.items() for v in seg]))
        else:
            return None
    return plans

def test_insert_path(plans: dict, target_plan_index: int, arc: tuple, G: nx.DiGraph, hubs: dict, start_node: int, dest_node: int, goal = 'Local Search'):
    free_hubs = get_free_hubs(plans, hubs)
    if plans[target_plan_index].nodes:
        subgraph_nodes = copy.deepcopy([v for v in plans[target_plan_index].nodes if v in hubs and count_hub_usage(plans[target_plan_index], v) < hubs[v][1]])
    else:
        subgraph_nodes = []
    # G_aux = nx.subgraph(G, subgraph_nodes + [start_node, dest_node] + list(free_hubs.keys()))
    G_aux = nx.subgraph(G, subgraph_nodes + [start_node, dest_node])
    if not start_node in hubs:
            subgraph_nodes.append(start_node)
    if not dest_node in hubs:
            subgraph_nodes.append(dest_node)

    G_aux = extend_auxiliary_graph_by_free_hubs(G, G_aux, free_hubs, hubs)

    if nx.has_path(G_aux, source=start_node, target=dest_node):
        new_segment = []
        if 'Local Search' == goal:
            new_segment = nx.shortest_path(G_aux, source=start_node, target=dest_node, weight='weight')
        else:
            new_segment = nx.shortest_path(G_aux, source=start_node, target=dest_node)
            # new_segment = get_longest_path(G_aux, start_node, dest_node)
        new_segment = remove_cost_edges(new_segment)
        plans[target_plan_index].segments[arc] = new_segment
        plans[target_plan_index].nodes = list(set([v for arc, seg in plans[target_plan_index].segments.items() for v in seg]))
        # plans = remove_cycles(plans)
        return plans
    else:
        return None

def get_used_real_arcs(plan: dict):
    used_real_arcs = []
    for arc, seg in plan.items():
        for i in range(len(seg)-1):
            if not (seg[i], seg[i+1]) in used_real_arcs:
                used_real_arcs.append((seg[i], seg[i+1]))
    return used_real_arcs

def get_plans_from_solution_adaptive(start_solution: dict, source_v: int, dest_v: int, plans_number: int):
    plans = {i : Plan(i, source_v, dest_v) for i in range(plans_number)}

    for key, segs in start_solution.items():
        for arc, seg in segs:
            plans[key].add_segment(arc, seg)
    return plans

def has_plan_other_nodes(plan: dict, nodes: list, source_v: int, dest_v: int):
    # print(f'{nodes}')
    # print(f'{plan.values()}')
    for arc, seg in plan.items():
        for v in seg:
            if v != source_v and v != dest_v and v in nodes:
                return True
    return False