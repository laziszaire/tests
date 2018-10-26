__author__ = "Li Tao, ltipchrome@gmail.com"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Node:
    """
    linked list node
    """
    def __init__(self, val=None):
        self.val = val
        self.next = None

    def __repr__(self):
        return f'Node({self.val})'

    def __bool__(self):
        return self.val is not None


class LinkedList:
    def __init__(self, head=None):
        self.head = head
        self.size = 0

    def append(self, val):
        _node = Node(val)
        cur = self.head
        if cur is None: self.head = _node
        else:
            while (cur is not None) and (cur.next is not None):
            # while cur and cur.next:
                cur = cur.next
            cur.next = _node
        self.size += 1

    def remove(self, val):
        prev = None
        cur = self.head
        while cur is not None:
            if cur.val == val:
                if prev is not None:
                    prev.next = cur.next
                else:
                    self.head = cur.next
                self.size -= 1
                return True
            prev = cur
            cur = cur.next

        return False

    def __repr__(self):
        return f'LinkedList(head={self.head})'

    def __len__(self):
        return self.size

    def __bool__(self):
        return self.size > 0

    def __contains__(self, val):
        cur = self.head
        while cur is not None:
            if cur.val == val:
                return True
            cur = cur.next
        return False

    def __iter__(self):
        cur = self.head
        while cur and cur.val:
            yield cur.val
            cur = cur.next

    def __str__(self):
        return str(list(self))


def test_LinkedList():
    a = [1, 2, 3]
    l = LinkedList()
    assert bool(l) is False
    for e in a:
        l.append(e)
    assert len(l) == len(a)
    print(l)
    assert (3 in l) is True
    l.remove(3)
    assert (3 in l) is False
    print(l)
    print(f'length of l: {len(l)}')


if __name__ == '__main__':
    test_LinkedList()

