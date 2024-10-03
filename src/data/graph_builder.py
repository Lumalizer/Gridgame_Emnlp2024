import torch
from data.graphstring_builder import GraphstringBuilder
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt



@dataclass
class GraphBuilder(GraphstringBuilder):
    embedding_size: int = None
    node_encoding: str = 'functor'

    def __post_init__(self):
        super().__post_init__()
        assert self.embedding_size is not None
        # generate integer encodings for shapes and position types
        self.pos_codes = {'above': [1, 0, 0, 0], 'below': [0, 1, 0, 0], 'left': [0, 0, 1, 0], 'right': [0, 0, 0, 1]}
        self.shapes_codes = {s: i+1 for i, s in enumerate(sorted(self.shapes))}
        self.shapes_codes['origin'] = 0
        self.shapes_codes['0'] = 0

    def get_node_elements(self, graphstring: str, add_empties):
        graphstring = graphstring.replace('.png', '')
        graphelements = graphstring.split('_')
        elements = []

        for i, shape in enumerate(graphelements):
            if not add_empties and shape == '0':
                continue
            node_index = len(elements)
            updown = self.pos_codes['above'] if i < 2 else self.pos_codes['below']
            leftright = self.pos_codes['right'] if i % 2 else self.pos_codes['left']
            positions = [updown[i] + leftright[i] for i in range(4)]
            elements.append((shape, self.shapes_codes[shape], node_index, i, positions))

        return elements

    def build_edgeattr_graph(self, graphstring: str, add_empties=False):
        # positions as edge attributes
        node_elements = self.get_node_elements(graphstring, add_empties)
        nodes = []
        edges = []
        edge_attrs = []

        for shape1, shapeid1, index1, gridpos1, pos1 in node_elements:
            node = [0] * (len(self.shapes) + 1)
            node[shapeid1] = 1
            nodes.append(node)

            for shape2, shapeid2, index2, gridpos2, pos2 in node_elements:
                if gridpos1 == gridpos2:
                    continue

                edges.append((index1, index2))
                edge_attrs.append([pos1[i] - pos2[i] for i in range(4)])

        x = torch.tensor(nodes, dtype=torch.long)
        edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
        edge_attr = torch.tensor(edge_attrs).float()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def build_functor_graph(self, graphstring: str, add_empties=False):
        # bunny --> upleft --> rabbit ; rabbit --> downright --> bunny
        node_elements = self.get_node_elements(graphstring, add_empties)
        nodes = []
        edges = []

        j = 0

        for shape1, shapeid1, index1, gridpos1, pos1 in node_elements:
            node = [0] * (len(self.shapes) + 1) + [0] * 4
            node[shapeid1] = 1
            nodes.append(node)

        for shape1, shapeid1, index1, gridpos1, pos1 in node_elements:
            for shape2, shapeid2, index2, gridpos2, pos2 in node_elements:
                if gridpos1 == gridpos2:
                    continue

                pos_node_idx = len(node_elements) + j
                j += 1

                pos_node = [0] * (len(self.shapes) + 1) + [pos1[i] - pos2[i] for i in range(4)]
                nodes.append(pos_node)
                edges.append((index1, pos_node_idx))
                edges.append((pos_node_idx, index2))

        x = torch.Tensor(nodes)
        edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        return data

    def build_leaves_graph(self, graphstring: str, add_empties=False):
        # topleft --> bunny <--> rabbit <-- bottomright
        node_elements = self.get_node_elements(graphstring, add_empties)
        nodes = []
        edges = []

        j = 0

        for shape1, shapeid1, index1, gridpos1, pos1 in node_elements:
            node = [0] * (len(self.shapes) + 1) + [0] * 4
            node[shapeid1] = 1
            nodes.append(node)

        for shape1, shapeid1, index1, gridpos1, pos1 in node_elements:
            for shape2, shapeid2, index2, gridpos2, pos2 in node_elements:
                if gridpos1 == gridpos2:
                    continue

                pos_node_idx = len(node_elements) + j
                j += 1

                pos_node = [0] * (len(self.shapes) + 1) + pos2
                nodes.append(pos_node)
                edges.append((index1, pos_node_idx))
                edges.append((pos_node_idx, index1))

        edges.append((0, 1))
        edges.append((1, 0))

        x = torch.Tensor(nodes)
        edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        return data

    def build_posattr_graph(self, graphstring: str, add_empties: bool = False):
        # (bunny_topleft) <--> (rabbit_bottomright)
        node_elements = self.get_node_elements(graphstring, add_empties)
        nodes = []
        edges = []

        j = 0

        for shape1, shapeid1, index1, gridpos1, pos1 in node_elements:
            shape = [0] * (len(self.shapes) + 1)
            shape[shapeid1] = 1
            node = shape + pos1
            nodes.append(node)

        for shape1, shapeid1, index1, gridpos1, pos1 in node_elements:
            for shape2, shapeid2, index2, gridpos2, pos2 in node_elements:
                if gridpos1 == gridpos2:
                    continue

                edges.append((index1, index2))

        x = torch.Tensor(nodes)
        edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        return data

    def get_batched_data(self, datastrings=None):
        if datastrings is None:
            datastrings = self.datastrings

        if self.node_encoding == 'functor':
            f = self.build_functor_graph
        elif self.node_encoding == 'leaves':
            f = self.build_leaves_graph
        elif self.node_encoding == 'edgeattr':
            f = self.build_edgeattr_graph
        elif self.node_encoding == 'posattr':
            f = self.build_posattr_graph
        else:
            raise ValueError(
                f"Node encoding {self.node_encoding} not supported. Try from ['functor', 'position', 'edgeattr', 'posattr']")

        return Batch.from_data_list([f(graphstring) for graphstring in datastrings])

   
    def visualize_graph(self, data: Data, encoding=None):
        # visualize graph
        # make sure to show nodes, node attributes, edges, edge attributes

        shapes_inverted = {v: k for k, v in self.shapes_codes.items()}
        positions = {51: 'above', 52: 'below', 53: 'left', 54: 'right'}

        nodes = []

        for node in data.x:
            node = node.tolist()
            index = node.index(1)

            try:
                extra = node[index+1:].index(1)
                extra = '\n (plus_pos_info)'
            except ValueError:
                extra = ''

            if index in shapes_inverted.keys():
                nodes.append(shapes_inverted.get(index)+extra)
            else:
                nodes.append(positions.get(index))

        node_labels = {i: n for i, n in enumerate(nodes)}

        plt.figure()
        plt.title(f"Graph encoding: {encoding}")
        G = to_networkx(data)
        nx.draw(G, labels=node_labels, with_labels=True)

        if data.edge_attr is not None:
            edge_labels = {(u.item(), v.item()): data.edge_attr[i].tolist() for i, (u, v) in enumerate(data.edge_index.T)}
            print(edge_labels)
            nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), edge_labels=edge_labels)

        plt.show()

    def visualize_all_graphtypes(self):
        graphstring = "bunny_0_cat_0"
        encodings = {'functor': self.build_functor_graph, 'leaves': self.build_leaves_graph,
                     'edgeattr': self.build_edgeattr_graph, 'posattr': self.build_posattr_graph}

        for encoding, f in encodings.items():
            data = f(graphstring)
            self.visualize_graph(data, encoding)

    def visualize_embeddings(self, data):
        from sklearn.manifold import TSNE
        import plotly.express as px
        tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
        data = tsne.fit_transform(data)
        fig = px.scatter(x=data[:, 0], y=data[:, 1], color=self.datastrings)
        fig.show()
