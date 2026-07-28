import type { LaidOutNode, TreeNode } from "./types";

interface MutableLayout {
  id: string;
  node: TreeNode;
  x: number;
  y: number;
  parentId: string | null;
  children: MutableLayout[];
}

function toMutable(node: TreeNode, parentId: string | null): MutableLayout {
  const base: MutableLayout = {
    id: node.id,
    node,
    x: 0,
    y: 0,
    parentId,
    children: [],
  };
  if (node.kind === "decision") {
    base.children = [
      toMutable(node.left, node.id),
      toMutable(node.right, node.id),
    ];
  }
  return base;
}

/** Reingold–Tilford-ish tidy layout for binary trees (simplified). */
export function layoutTree(
  root: TreeNode,
  width: number,
  height: number,
  padding = 40,
): LaidOutNode[] {
  const tree = toMutable(root, null);
  const nodes: MutableLayout[] = [];

  function collect(n: MutableLayout) {
    nodes.push(n);
    n.children.forEach(collect);
  }
  collect(tree);

  const maxDepth = Math.max(...nodes.map((n) => n.node.depth), 0);
  const leafGap = 1;

  let nextX = 0;
  function firstWalk(n: MutableLayout) {
    if (n.children.length === 0) {
      n.x = nextX;
      nextX += leafGap;
      return;
    }
    n.children.forEach(firstWalk);
    n.x = (n.children[0].x + n.children[n.children.length - 1].x) / 2;
  }
  firstWalk(tree);

  const minX = Math.min(...nodes.map((n) => n.x));
  const maxX = Math.max(...nodes.map((n) => n.x));
  const span = Math.max(maxX - minX, 1);
  const usableW = width - padding * 2;
  const usableH = height - padding * 2;

  return nodes.map((n) => ({
    id: n.id,
    node: n.node,
    parentId: n.parentId,
    x: padding + ((n.x - minX) / span) * usableW,
    y:
      padding +
      (maxDepth === 0 ? usableH / 2 : (n.node.depth / maxDepth) * usableH),
  }));
}

export function countNodes(root: TreeNode): number {
  if (root.kind === "leaf") return 1;
  return 1 + countNodes(root.left) + countNodes(root.right);
}

export function treeDepth(root: TreeNode): number {
  if (root.kind === "leaf") return 0;
  return 1 + Math.max(treeDepth(root.left), treeDepth(root.right));
}
