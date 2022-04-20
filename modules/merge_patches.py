import torch

#Assume that children of the same node at the bottom of the tree are geometrically adjacent.

class Node:
    def __init__(self, points, u):
        self.parent = None
        self.left = None
        self.right = None
        self.points = points
        self.u = u
        self.merged = False

def merge(root):
    while root.left != None:
        leaves = []
        find_leaves(root, leaves)
        flatten(leaves)
        for leaf in leaves:
            if leaf.parent != None and self.merged:
                leaf.parent = Node(leaf.points, leaf.u)

def find_leaves(node, leaves):
    #check that node is not a leaf
    if node.left == None:
        leaves.append(node)
    else:
    #node is parent of leaves
    if node.left.u != None and node.right.u != None:
        leaves.append(merge_leaves(node.left, node.right))
    elif node.left.u != None:
        leaves.append(node.left)
        find_leaves(node.right, leaves)
    elif node.right.u != None:
        leaves.append(node.right)
        find_leaves(node.left, leaves)
    find_leaves(node.left, leaves)
    find_leaves(node.right, leaves)

def merge_leaves(leaf1, leaf2):
    u = leaf1.u+leaf2.u if torch.dot(leaf1.u, leaf2.u) < 0 else leaf1.u-leaf2.u
    u = u/torch.linalg.norm(u)

    points = leaf1.points.append(leaf2.points)

    node = Node(points, u)
    node.merged = True

    return node
