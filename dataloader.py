import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import dgl
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    smiles_to_bigraph,
)

from utils import integer_label_protein

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _progress(iterable, total=None, desc=None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def extract_k_hop_subgraph_dgl(graph, root_node, k):
    src, dst = graph.edges()
    visited = {root_node}
    current_level = [root_node]

    for _ in range(k):
        next_level = []
        for node in current_level:
            out_neighbors = dst[src == node].unique().tolist()
            in_neighbors = src[dst == node].unique().tolist()
            neighbors = list(set(out_neighbors + in_neighbors))

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.append(neighbor)
        current_level = next_level
        if not current_level:
            break

    subgraph_nodes = sorted(list(visited))
    node_ids = torch.tensor(subgraph_nodes, dtype=torch.int32)
    subgraph = graph.subgraph(node_ids)
    return subgraph, subgraph_nodes


def create_nested_subgraphs_dgl(graph, h=2, max_nodes_per_hop=None):
    num_nodes = graph.num_nodes()
    subgraphs = []
    subgraph_node_counts = []

    for root_node in range(num_nodes):
        subgraph, _ = extract_k_hop_subgraph_dgl(graph, root_node, h)
        subgraphs.append(subgraph)
        subgraph_node_counts.append(subgraph.num_nodes())

    batched_graph = dgl.batch(subgraphs)

    node_to_subgraph = []
    current_subgraph = 0
    for count in subgraph_node_counts:
        node_to_subgraph.extend([current_subgraph] * count)
        current_subgraph += 1
    node_to_subgraph = torch.tensor(node_to_subgraph, dtype=torch.long)

    subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)
    batched_graph.ndata["node_to_subgraph"] = node_to_subgraph
    batched_graph.subgraph_to_graph = subgraph_to_graph
    batched_graph.num_subgraphs = len(subgraphs)

    return batched_graph, node_to_subgraph, subgraph_to_graph


def _build_dti_sample(smiles, protein_seq, label, max_drug_nodes=290, use_nested=True, h=2):
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
    graph_builder = partial(smiles_to_bigraph, add_self_loop=True)

    v_d = graph_builder(
        smiles=smiles,
        node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer,
    )

    actual_node_feats = v_d.ndata.pop("h")
    num_actual_nodes = actual_node_feats.shape[0]
    num_virtual_nodes = max_drug_nodes - num_actual_nodes
    virtual_node_bit = torch.zeros([num_actual_nodes, 1])
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
    v_d.ndata["h"] = actual_node_feats

    virtual_node_feat = torch.cat(
        (torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)),
        1,
    )
    v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
    v_d = v_d.add_self_loop()

    if use_nested:
        v_d, node_to_subgraph, subgraph_to_graph = create_nested_subgraphs_dgl(v_d, h=h)
        v_d.ndata["node_to_subgraph"] = node_to_subgraph
        v_d.subgraph_to_graph = subgraph_to_graph
        v_d.num_subgraphs = len(subgraph_to_graph)

    protein = integer_label_protein(protein_seq)
    return v_d, protein, label


def _build_dti_sample_from_record(record, max_drug_nodes=290, use_nested=True, h=2):
    return _build_dti_sample(
        smiles=record["SMILES"],
        protein_seq=record["Protein"],
        label=record["Y"],
        max_drug_nodes=max_drug_nodes,
        use_nested=use_nested,
        h=h,
    )


def build_cache_samples(
    df,
    max_drug_nodes=290,
    use_nested=True,
    h=2,
    num_workers=1,
    worker_type="process",
    show_progress=False,
    progress_desc="Building cache",
):
    records = df[["SMILES", "Protein", "Y"]].to_dict("records")

    if num_workers <= 1:
        iterator = (
            _build_dti_sample_from_record(
                record,
                max_drug_nodes=max_drug_nodes,
                use_nested=use_nested,
                h=h,
            )
            for record in records
        )
        if show_progress:
            iterator = _progress(iterator, total=len(records), desc=progress_desc)
        return list(iterator)

    executor_cls = ProcessPoolExecutor if worker_type == "process" else ThreadPoolExecutor
    cached_samples = [None] * len(records)

    with executor_cls(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _build_dti_sample_from_record,
                record,
                max_drug_nodes,
                use_nested,
                h,
            ): idx
            for idx, record in enumerate(records)
        }

        completed = as_completed(futures)
        if show_progress:
            completed = _progress(completed, total=len(futures), desc=progress_desc)

        for future in completed:
            idx = futures[future]
            cached_samples[idx] = future.result()

    return cached_samples


class DTIDataset(data.Dataset):
    def __init__(
        self,
        list_IDs,
        df,
        max_drug_nodes=290,
        use_nested=True,
        h=2,
        dataset_name=None,
        split_name=None,
        split_file_name=None,
        cache_root="./datasets/subgraph_cache",
    ):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.use_nested = use_nested
        self.h = h
        self.dataset_name = dataset_name or "unknown_dataset"
        self.split_name = split_name or "unknown_split"
        self.split_file_name = split_file_name or "data"
        self.cache_root = cache_root

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.cached_samples = self._load_or_build_cache()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        return self.cached_samples[index]

    def _get_cache_path(self):
        cache_dir = os.path.join(
            self.cache_root,
            self.dataset_name,
            self.split_name,
            f"hop_{self.h}",
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.split_file_name}.pt")

    def _build_sample(self, index):
        index = self.list_IDs[index]
        return _build_dti_sample(
            smiles=self.df.iloc[index]["SMILES"],
            protein_seq=self.df.iloc[index]["Protein"],
            label=self.df.iloc[index]["Y"],
            max_drug_nodes=self.max_drug_nodes,
            use_nested=self.use_nested,
            h=self.h,
        )

    def _load_or_build_cache(self):
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            print(f"Loading cached subgraphs from {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        print(f"Building subgraph cache: {cache_path}")
        cached_samples = [self._build_sample(i) for i in range(len(self.list_IDs))]
        torch.save(cached_samples, cache_path)
        print(f"Saved subgraph cache to {cache_path}")
        return cached_samples


def collate_fn_nested(batch):
    graphs, proteins, labels = zip(*batch)

    has_subgraph_info = (
        isinstance(graphs[0], dgl.DGLGraph)
        and "node_to_subgraph" in graphs[0].ndata
    )

    if has_subgraph_info:
        batched_graph = dgl.batch(graphs)

        node_to_subgraph_list = []
        subgraph_to_graph_list = []
        current_subgraph_offset = 0
        current_graph_id = 0

        for graph in graphs:
            if hasattr(graph, "num_subgraphs"):
                num_subgraphs = graph.num_subgraphs
            elif "node_to_subgraph" in graph.ndata:
                num_subgraphs = graph.ndata["node_to_subgraph"].max().item() + 1
            else:
                num_subgraphs = 1

            if "node_to_subgraph" in graph.ndata:
                node_to_subgraph_list.append(
                    graph.ndata["node_to_subgraph"] + current_subgraph_offset
                )
            else:
                node_to_subgraph_list.append(
                    torch.zeros(graph.num_nodes(), dtype=torch.long)
                    + current_subgraph_offset
                )

            if hasattr(graph, "subgraph_to_graph") and graph.subgraph_to_graph is not None:
                subgraph_to_graph_list.append(
                    graph.subgraph_to_graph + current_graph_id
                )
            else:
                subgraph_to_graph_list.append(
                    torch.zeros(num_subgraphs, dtype=torch.long) + current_graph_id
                )

            current_subgraph_offset += num_subgraphs
            current_graph_id += 1

        batched_graph.ndata["node_to_subgraph"] = torch.cat(node_to_subgraph_list)
        batched_graph.subgraph_to_graph = torch.cat(subgraph_to_graph_list)
        batched_graph.num_subgraphs = current_subgraph_offset
    else:
        batched_graph = dgl.batch(graphs)

    if isinstance(proteins[0], torch.Tensor):
        proteins = torch.stack(proteins)
    else:
        proteins = torch.tensor(proteins)

    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)
    else:
        labels = torch.tensor(labels)

    return batched_graph, proteins, labels


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
