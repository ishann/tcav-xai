import json

import pandas as pd


class Hierarchy:
    def __init__(self, json_path='hierarchy.json'):
        with open(json_path, 'r') as f:
            self.json = json.load(fp=f)
        self.validate_json()

    def get_leaf_nodes(self):
        leafs = []
        frontier = [self.json]

        while len(frontier) > 0:
            node = frontier.pop()

            # Check if node is leaf
            if 'children' not in node or node['children'] is None or len(node['children']) == 0:
                leafs.append(node['name'])
            else:
                for c in node['children']:
                    frontier.append(c)
        return leafs

    def validate_json(self, imagenet_url_map_path='imagenet_url_map.csv'):
        df = pd.read_csv(imagenet_url_map_path)
        leaf_classes = set(df['class_name'].tolist())

        frontier = [self.json]

        while len(frontier) > 0:
            node = frontier.pop()

            assert ('name' in node)

            # Check if node is leaf
            if 'children' not in node or node['children'] is None or len(node['children']) == 0:
                assert (node['name'] in leaf_classes)
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


if __name__ == '__main__':
    h = Hierarchy()
    print(h.get_leaf_nodes())
