import qelos as q


class Namespace(str):
    """
    Namespace for AllenNLP preprocessing.
    """
    pass


class NS(Namespace): pass


class IndexKey(str):
    """
    For index keys in AllenNLP fields and BasicTextFieldEmbedders
    """
    pass


class IK(IndexKey): pass


class FieldName(str):
    """
    For field names in AllenNLP instances
    """
    pass


class FK(FieldName): pass


if __name__ == '__main__':
    x = NS("dummy")
    print(x)
    print(len(x))
    x