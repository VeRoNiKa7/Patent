from copy import deepcopy
from dataclasses import dataclass, astuple
from itertools import chain
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import defaultdict

DEL_ROLES = ('aux', 'auxpass', 'cc', 'det', 'predet',
             'prep', 'prt', 'punct', 'quantmod', 'tmod', 'mark')

ROLE_MAP = {
    'ATTR': ('acop', 'amod', 'arg', 'attr', 'cop', 'compound',
             'expl', 'mod', 'neg', 'nn', 'poss', 'possessive'),
    'APPEND': ('appos', 'num', 'number', 'ref', 'sdep'),
    'COORD': ('11mark', 'acomp', 'advcl', 'advmod', 'ccomp', 'comp', 'infmod',
              'mwe', 'npadvmod', 'parataxis', 'partmod', 'pcomp', 'rcmod', 'xcomp'),
    'I': ('agent', 'csubj', 'csubjpass', 'nsubj', 'nsubjpass', 'subj', 'xsubj'),
    'II': ('dobj', 'iobj', 'obj', 'pobj'),
    'CONJ': ('conj', 'preconj')
}

class RemoveRootError(Exception):
    pass

class Container:
    def __init__(self, data):
        self.__dict__['__data__'] = data

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if '__data__' in self.__dict__:
            if item in self.__dict__['__data__'].__dict__:
                return self.__dict__['__data__'].__dict__[item]
        raise AttributeError

    def __setattr__(self, key, value):
        if key in self.__data__.__dict__:
            self.__data__.__dict__[key] = value
        self.__dict__[key] = value

    def __repr__(self):
        return f'{self.__class__.__name__}(data={repr(self.__data__)})'

class Tree(Container):
    def __init__(self, data, parent=None):
        super().__init__(data)
        self.parent = parent
        self._children = []

    def add_child(self, data):
        if not isinstance(data, type(self.__data__)):
            raise TypeError

        child = self.__class__(data, self)
        self._children.append(child)
        return child

    def remove(self):
        if self.parent is None:
            raise RemoveRootError
        self.parent._children.remove(self)
        for child in self._children:
            child.parent = self.parent
            self.parent._children.append(child)

    def children(self, key=None):
        yield from filter(key, self._children)

    def __iter__(self):
        return chain((self,), *map(iter, self._children))

    def __len__(self):
        return sum(1 for _ in self)

@dataclass
class Conll:
    id: int
    form: str
    lemma: str
    upostag: str
    xpostag: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str

    @classmethod
    def _from_string(cls, string):
        string = string.strip()
        fields = string.split('\t')

        if len(fields) != 10:
            raise ValueError

        fields[0] = int(fields[0])
        fields[6] = int(fields[6])
        return cls(*fields)

    @classmethod
    def parse(cls, string):
        string = string.strip()
        rows = string.split('\n')
        yield from map(cls._from_string, rows)

    def __str__(self):
        fields = (str(f) for f in astuple(self))
        return '\t'.join(fields)

class ConllTree(Tree):
    def remove(self):
        super().remove()
        for child in self.parent._children:
            child.head = self.parent.id

    def __str__(self):
        return '\n'.join(str(node.__data__) for node in self)

    def __iter__(self):
        yield from sorted(super().__iter__(), key=lambda x: x.id)

    @classmethod
    def _from_list(cls, items):
        def _fill(root, items):
            children = filter(lambda x: x.head == root.id, items)
            for child in children:
                _fill(root.add_child(child), items)

        items = list(items)
        root = next(filter(lambda x: x.head == 0, items))
        tree = cls(root)
        _fill(tree, items)
        return tree

    @classmethod
    def _from_string(cls, string):
        string = string.strip()
        rows = string.split('\n')
        rows = filter(lambda x: not x.startswith('#'), rows)
        rows = map(Conll._from_string, rows)
        return cls._from_list(rows)

    @classmethod
    def parse(cls, string):
        string = string.strip()
        blocks = string.split('\n' * 2)
        yield from map(cls._from_string, blocks)

class SAO:
    def __init__(self, tree):
        self._tree = tree

    @property
    def subjects(self):
        return filter(lambda x: x.deprel == 'I', self._tree)

    @property
    def action(self):
        return self._tree

    @property
    def objects(self):
        return filter(lambda x: x.deprel == 'II', self._tree)

    def compare(self, other):
        def _compare_attrs(node1, node2):
            def _attrs(node):
                children = node.children(lambda x: x.deprel == 'ATTR')
                return list(children)

            attr1, attr2 = map(_attrs, (node1, node2))

            count = max(map(len, (attr1, attr2)))
            if count == 0:
                return 1
            matches = sum(x in attr2 for x in attr1)
            return matches / count

        def _compare_nodes(node_list1, node_list2):
            def _compare(node_list1, node_list2):
                for i in node_list1:
                    max_k = 0
                    match = 0
                    for j in node_list2:
                        if i.lemma == j.lemma:
                            match = 1
                            k = _compare_attrs(i, j)
                            max_k = max(k, max_k)
                    yield max_k + match

            node_list1, node_list2 = map(list, (node_list1, node_list2))

            result = _compare(node_list1, node_list2)

            return sum(result) / max(map(len, (node_list1, node_list2)))

        if self._tree.lemma != other._tree.lemma:
            return 0

        k_action = _compare_attrs(self._tree, other._tree)
        k_subjects = _compare_nodes(
            self._tree.children(lambda x: x.deprel == 'I'),
            other._tree.children(lambda x: x.deprel == 'I')
        )
        k_objects = _compare_nodes(
            self._tree.children(lambda x: x.deprel == 'I'),
            other._tree.children(lambda x: x.deprel == 'I')
        )
        return sum((k_action, k_subjects, k_objects)) / 5

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        nodes = sorted(self._tree, key=lambda x: (x.head, x.lemma))
        lemmas = tuple(x.lemma for x in nodes)
        return hash(lemmas)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        def _lemmas(items): return ', '.join(f'"{x.lemma}"' for x in items)
        template = '{class_name}(subjects=({subjects}), action="{action}", objects=({objects}))'
        template = template.format(
            class_name=self.__class__.__name__,
            subjects=_lemmas(self.subjects),
            action=self.action.lemma,
            objects=_lemmas(self.objects),
        )
        return template

    @staticmethod
    def _reduce(tree):
        tree = deepcopy(tree)

        for node in tree:
            if node.deprel in DEL_ROLES:
                node.remove()

        for node in tree:
            cur_role = node.deprel
            for new_role, old_roles in ROLE_MAP.items():
                if cur_role in old_roles:
                    node.deprel = new_role

        return tree

    @classmethod
    def extract(cls, tree):
        reduced_tree = cls._reduce(tree)
        verb_indices = list(x.id for x in reduced_tree if x.upostag == 'VERB')
        for index in verb_indices:
            tree = deepcopy(reduced_tree)
            verb = next(filter(lambda x: x.id == index, tree))
            subjects = list(verb.children(lambda x: x.deprel == 'I'))
            objects = list(verb.children(lambda x: x.deprel == 'II'))

            if verb.id != 0 and len(subjects) == 0:
                grandparent = verb.parent
                if grandparent:
                    subjects = list(grandparent.children(lambda x: x.deprel == 'II'))
                    for subj in subjects:
                        subj.deprel = 'I'
                        subj.head = verb.id

            if not len(subjects) or not len(objects):
                continue

            verb.head = 0
            verb.deprel = 'root'

            selected_nodes = [verb, *subjects, *objects]
            attr_nodes = []
            for node in selected_nodes:
                attr_children = node.children(lambda x: x.deprel == 'ATTR')
                conj_children = list(node.children(lambda x: x.deprel == 'CONJ'))
                for child in conj_children:
                    if child.parent.deprel == 'I':
                        child.deprel = 'I'
                    else:
                        child.deprel = 'II'
                attr_nodes.extend(attr_children)
                attr_nodes.extend(conj_children)

            selected_nodes.extend(attr_nodes)

            # FIXME: x.__data__
            selected_nodes = list(deepcopy(x.__data__) for x in selected_nodes)
            yield SAO(ConllTree._from_list(selected_nodes))


def sao_distance(a, b):
    return 1 - a.compare(b)

#Сравнивает все полученные SAO между собой
def matching(iterable, key, t):
    def _flat(func):
        def _wrapper(a, b):
            a, = a
            b, = b
            return func(a, b)
        return _wrapper
    key = _flat(key)
    iterable = [[x] for x in iterable]
    dicts = pdist(iterable, key)
    dict_mtx = linkage(dicts, method='complete')
    labels = fcluster(dict_mtx, t, criterion='distance')
    iterable = chain(*iterable)
    matches = defaultdict(list)
    for label, sao in zip(labels, iterable):
        matches[label].append(sao)
    return {k - 1: v for k, v in matches.items()}