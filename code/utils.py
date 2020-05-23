import copy
import pandas as pd
import numpy as np
import networkx as nx
from networkx.convert_matrix import from_pandas_edgelist

## -- Parameters -- ##
HH_LABELS = ['Family Member',
            'Family member',
            'Daughter - Mother',
            'Father - Son',
            'Husband - Wife',
            'Close Contact'
            'Works in same household']

## -- General Utils -- ##
def merge_lists(list_of_lists, unique = True):
    merged = ([item for sublist in list_of_lists for item in sublist])
    if unique:
        merged = list(set(merged))
    return merged
def binomial_ci(p,n):
    return(p -1.96*np.sqrt(p*(1-p)/n), p + 1.96*np.sqrt(p*(1-p)/n))

def compute_sd(individual_rh, total_sources):
    padded = np.pad(individual_rh, (0, total_sources), 'constant')
    return(np.std(padded)/np.sqrt(total_sources))

## -- Data Loading and Cleaning Utils -- ##
def load_case_data(fname='../data/case_data_raw.csv'):
    # Load Data
    case_df = pd.read_csv(fname)
    # Data Preprocessing
    case_df =  case_df.replace('-', np.nan)
    numeric_cols = ['Age','Symptomatic ToConfirmation','Days ToRecover']
    datetime_cols = ['Symptomatic At', 'Confirmed At', 'Recovered At']
    case_df[numeric_cols] = pd.to_numeric(case_df[numeric_cols].stack()).unstack()
    case_df[datetime_cols] = pd.to_datetime(case_df[datetime_cols].stack()).unstack()
    case_df['Case'] = case_df['Case'].astype('str')
    # Add Relevant Columns
    symptomatic_to_recovery_col = 'SymptomaticToRecovery'
    case_df[symptomatic_to_recovery_col] = case_df['Days ToRecover'] + case_df['Symptomatic ToConfirmation']
    return case_df

def load_contact_data(fname = '../data/contact_data_raw.csv', hh_labels = HH_LABELS):
    contact_df = pd.read_csv(fname)
    contact_df = contact_df.drop_duplicates()
    # Add Relevant Columns
    contact_df['isClusterTarg'] = contact_df.targ.str.startswith(('Cluster'))
    contact_df['isClusterSrc'] = contact_df.src.str.startswith(('Cluster'))
    contact_df['isCluster'] = contact_df['isClusterTarg'] | contact_df['isClusterSrc']
    contact_df['isHH'] = contact_df['label'].isin(hh_labels)
    return contact_df

def get_infection_graph(clean = True, delta_days = 5, case_df=None, contact_df=None):
    # Load case and contact data
    if case_df is None:
        case_df = load_case_data()
    if contact_df is None:
        contact_df = load_contact_data()
    # Create directed graph
    G_directed = from_pandas_edgelist(df=contact_df, source='src', target='targ',
                        edge_attr=list(contact_df.columns.drop(['src', 'targ'])),
                        create_using=nx.DiGraph())
    # Assign all data from case_df as node attributes per case
    nx.set_node_attributes(G_directed, case_df.set_index('Case').to_dict('index'))
    if clean==False:
        return G_directed

    # Correct wrongly pointing arrow whenever possible
    G_corrected = copy.deepcopy(G_directed)

    all_nodes = G_directed.nodes(data=True)
    all_edges = G_directed.edges()

    # for every edge check that 'Symptomatic AtSrc' is before 'Symptomatic AtTrg', if not then switch
    # then for every edge check that 'Confirmed AtSrc' is 7 days before 'Confirmed AtTrg', if not then switch
    for e in all_edges:
        e_data = G_directed.get_edge_data(*e)
        if not pd.isnull(all_nodes[e[0]].get('Symptomatic At')): # <--src
            if not pd.isnull(all_nodes[e[1]].get('Symptomatic At')): # <--targ
                # if the target is symptomatic earlier then swap
                if all_nodes[e[1]].get('Symptomatic At') < all_nodes[e[0]].get('Symptomatic At'):
                    G_corrected.remove_edge(*e)
                    G_corrected.add_edge(e[1], e[0], **e_data)
        elif not pd.isnull(all_nodes[e[0]].get('Confirmed At')): # <--src
            if not pd.isnull(all_nodes[e[1]].get('Confirmed At')): # <--targ
                # if the target is tested positive days before the source then swap
                if all_nodes[e[1]].get('Confirmed At') < all_nodes[e[0]].get('Confirmed At')-pd.to_timedelta(delta_days, unit='days'):
                    G_corrected.remove_edge(*e)
                    G_corrected.add_edge(e[1], e[0], **e_data)
    G_directed = copy.deepcopy(G_corrected)
    return G_directed

## -- Graph utils -- ##

def is_household(edge_data):
    if edge_data.get('isHH') is not None:
        return edge_data.get('isHH')
    else:
        return edge_data.get('attr_dict').get('isHH')

def is_cluster(node_label):
    return node_label.startswith('Cluster')

def is_source(node_data, cut_off_dt):
    confirmed_dt = node_data.get('Confirmed At')
    if pd.isnull(confirmed_dt):
        return False
    else:
        return confirmed_dt <= cut_off_dt

def get_cluster_nodes(graph):
    all_nodes = graph.nodes()
    cluster_nodes = [node for node in iter(all_nodes) if is_cluster(node)]
    return cluster_nodes

def to_directed(graph, parent_digraph):
    return parent_digraph.subgraph(graph.nodes())

def get_household_connected_components(graph):
    is_directed = nx.is_directed(graph)
    household_edges = get_household_edges(graph)
    household_graph = graph.edge_subgraph(household_edges)
    if not is_directed:
        return nx.connected_component_subgraphs(household_graph)
    household_cc = nx.connected_component_subgraphs(household_graph.to_undirected(reciprocal=False))
    directed_household_cc = [to_directed(cc, household_graph) for cc in household_cc]
    return directed_household_cc

def get_primary_node(digraph, type = 'first_date'):
    def _get_primary_first_date(digraph):
        all_nodes = digraph.nodes(data = True)
        primary_date = pd.datetime.today()
        primary_node = []
        for (node, data) in all_nodes:
            conf_date = data.get('Confirmed At')
            if not pd.isnull(conf_date):
                if conf_date < primary_date:
                    primary_date = conf_date
                    primary_node = node
        return(primary_node)

    def _get_primary_max_out(digraph):
        all_nodes = digraph.nodes()
        primary_node = []
        max_successors = 0
        for node in all_nodes:
            successors = set(digraph.successors(node))
            if len(successors) > max_successors:
                max_successors = len(successors)
                primary_node = node
        return(primary_node)

    def _get_primary_max_degree(digraph):
        all_nodes = digraph.nodes()
        primary_node = []
        max_degree = 0
        for node in all_nodes:
            successors = set(digraph.successors(node))
            predecessors = set(digraph.successors(node))
            if (len(successors)+len(predecessors)) > max_degree:
                max_degree = (len(successors)+len(predecessors))
                primary_node = node
        return(primary_node)

    def _get_primary_no_in(digraph):
        all_nodes = digraph.nodes()
        for node in all_nodes:
            predecessors = set(digraph.successors(node))
            if len(predecessors) == 0:
                return(node)
        return []

    def _get_primary_min_id(digraph):
        all_nodes = digraph.nodes()
        primary_node = []
        min_id = 100000
        for node in all_nodes:
            try:
                num_node = int(node)
                if num_node < min_id:
                    min_id = num_node
                    primary_node = node
            except:
                continue
        return(primary_node)

    if type == 'first_date':
        return _get_primary_first_date(digraph)
    if type == 'max_out':
        return _get_primary_max_out(digraph)
    if type == 'max_degree':
        return _get_primary_max_degree(digraph)
    if type == 'no_in':
        return _get_primary_no_in(digraph)
    if type == 'min_id':
        return _get_primary_min_id(digraph)
    return []

def get_household_edges(graph):
    all_edges = graph.edges()
    household_edges = [e for e in iter(all_edges) if is_household(graph.get_edge_data(*e))]
    return(household_edges)

def get_source_cases(graph, cut_off_dt, case_df=None):
    graph_source_cases = [node for (node, node_data) in iter(graph.nodes(data=True)) if is_source(node_data, cut_off_dt)]
    singleton_cases = []
    if case_df is not None:
        all_source_cases = (case_df[case_df['Confirmed At'] <= cut_off_dt]).Case
        singleton_cases = list(set(all_source_cases).difference(set(graph_source_cases)))
    return dict(connected_sources = graph_source_cases, singleton_sources = singleton_cases)

def get_individual_rh(hh_component, source_nodes):
    hh_nodes = hh_component.nodes()
    hh_size = len(hh_nodes)
    r_h_vals = []
    for node in iter(hh_nodes):
        if node in source_nodes:
            r_h_vals.append(hh_size -1)
    return r_h_vals

def get_neighbors(nodes, graph, keep_cluster=True, directed=False, n_type='succ'):
    all_nodes = graph.nodes()
    # Check if there are node not contained in the graph
    extra_nodes = set(nodes).difference(all_nodes)
    if len(extra_nodes) > 0:
        print("Removing {} nodes from the list as they are not present in the graph".format(len(extra_nodes)))
        nodes = set(nodes).difference(extra_nodes)

    if directed == False:
        neighbor_nodes = [list(graph.neighbors(node)) for node in nodes]
    else:
        if n_type == 'succ':
            neighbor_nodes = [list(graph.successors(node)) for node in nodes]
        elif n_type == 'pred':
            neighbor_nodes = [list(graph.predecessors(node)) for node in nodes]
        else:
            neighbor_nodes = [list(graph.successors(node)) for node in nodes] + [list(graph.predecessors(node)) for node in nodes]

    neighbor_nodes = merge_lists(neighbor_nodes, unique=True)
    if not keep_cluster:
        neighbor_nodes = [node for node in neighbor_nodes if not is_cluster(node)]
    return(neighbor_nodes)

def print_stats(digraph):
    graph = digraph.to_undirected(reciprocal=False)
    connected_components = list(nx.connected_component_subgraphs(graph))
    print("The full graph has {} nodes, {} edges and {} connected components".format(len(graph.nodes()),
                                                                                len(graph.edges()),
                                                                                len(connected_components)
                                                                                ))
    n_nodes = [len(cc.nodes()) for cc in connected_components]
    print("The largest connected component contains {} nodes".format(max(n_nodes)))

    cluster_nodes = get_cluster_nodes(graph)
    print("Out of the {} nodes {} are cases and {} are clusters".format(len(graph.nodes()),
                                                                        len(graph.nodes())-len(cluster_nodes),
                                                                        len(cluster_nodes)
                                                                        ))
    cluster_neighbors = get_neighbors(cluster_nodes, graph)
    has_cluster = [len(get_cluster_nodes(cc)) > 0 for cc in connected_components]
    print("There are {} cases connected to clusters and {} of the connected components contain a cluster node".format(len(cluster_neighbors),
                                                                                                                 sum(has_cluster)))

    household_edges = get_household_edges(graph)
    has_household = [len(get_household_edges(cc)) > 0 for cc in connected_components]
    print("There are {} household edges and {} connected components contain household infections".format(len(household_edges),
                                                                                                        sum(has_household)))
