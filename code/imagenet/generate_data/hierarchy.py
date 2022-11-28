import json
from copy import deepcopy


def load_wordnet_labels(wordnet_path='wordnet_labels.txt'):
    labels_to_wordnet = dict()
    with open(wordnet_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if ": " not in line:
                continue
            line_arr = line.split(": ")
            wnid = line_arr[0]
            labels = line_arr[1].split(", ")

            for label in labels:
                labels_to_wordnet[label] = wnid
    return labels_to_wordnet


def load_imagenet_labels(imagenet_labels_path):
    with open(imagenet_labels_path, 'r') as f:
        idx2label = json.load(fp=f)

    # Invert the dict
    label2idx = dict()
    for idx, labels in idx2label.items():
        idx = int(idx)
        labels_arr = labels.split(", ")
        for label in labels_arr:
            label2idx[label] = idx
    return label2idx


class Hierarchy:
    def __init__(self, json_path='hierarchy.json', wordnet_labels_path='wordnet_labels.txt',
                 imagenet_idx_labels_path='imagenet1000_clsidx_to_labels.txt'):
        """
        :param json_path:
        :param wordnet_labels_path: Used to validate names and assign ids
        """
        with open(json_path, 'r') as f:
            self.json = json.load(fp=f)
        self.label2wordnet = load_wordnet_labels(wordnet_path=wordnet_labels_path)
        self.imagenet_label2idx = load_imagenet_labels(imagenet_labels_path=imagenet_idx_labels_path)
        self.validate_json()

    def get_leaf_nodes(self, of=None):
        """
        Get list of leaf node names of a node with specified name. If 'of' is blank, return all leaf nodes
        """
        leafs = []

        root = self.json

        if of:
            root = self.find_node(name=of)

        frontier = [root]

        while len(frontier) > 0:
            node = frontier.pop()

            # Check if node is leaf
            if 'children' not in node or node['children'] is None or len(node['children']) == 0:
                leafs.append(node['name'])
            else:
                for c in node['children']:
                    frontier.append(c)
        return leafs

    def validate_json(self):
        """
        Validate json and set node ids
        """
        leaf_classes = set(self.label2wordnet.keys())
        count = 0
        frontier = [self.json]

        while len(frontier) > 0:
            node = frontier.pop()

            assert 'name' in node

            if 'id' not in node:
                node['id'] = 1000 + count
                count += 1

            # Check if node is leaf
            if 'children' not in node or node['children'] is None or len(node['children']) == 0:
                assert node['name'] in leaf_classes
                node['id'] = self.imagenet_label2idx[node['name']]
            else:
                for c in node['children']:
                    frontier.append(c)

    def get_all_nodes(self):
        all = []
        frontier = [self.json]

        while len(frontier) > 0:
            node = frontier.pop()
            all.append(node['name'])

            # Check if node is leaf
            if not ('children' not in node or node['children'] is None or len(node['children']) == 0):
                for c in node['children']:
                    frontier.append(c)
        return list(set(all))

    def find_node(self, name):
        """
        Find node by name, returns node dict
        """
        frontier = [self.json]

        while len(frontier) > 0:
            node = frontier.pop()

            if node['name'] == name:
                return node

            # Check if node is leaf
            if not ('children' not in node or node['children'] is None or len(node['children']) == 0):
                for c in node['children']:
                    frontier.append(c)
        return None

    def get_path(self, name):
        """
        Return list of nodes on path to destination
        :param name: leaf node name
        :return: list
        """

        frontier = [self.json]
        path_frontier = [[self.json['name']]]
        while len(frontier) > 0:
            node = frontier.pop(0)
            path = path_frontier.pop(0)
            path.append(node['name'])

            if node['name'] == name:
                return path[1:]
            else:
                for c in node['children']:
                    frontier.append(c)
                    path_frontier.append(deepcopy(path))
                path.pop(-1)

    def get_children_names(self, name):
        """
        Get list of names of children of a node specified by name
        """
        node = self.find_node(name)
        return [n['name'] for n in node['children']]

if __name__ == '__main__':
    h = Hierarchy()
    print(h.get_leaf_nodes())
    print(h.get_path("dog"))

    print(h.get_children_names("dog"))
