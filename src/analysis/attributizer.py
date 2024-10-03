from typing import Union


class Attributizer:
    def __init__(self, shape_encoding, position_encoding):
        self.attrs = {}
        self.position_encoding = position_encoding
        self.shape_encoding = shape_encoding

        assert shape_encoding in ['none', 'entangled', 'disentangled']
        assert position_encoding in ['none', 'entangled', 'disentangled', 'double_per_node']
        assert not (shape_encoding == 'none' and position_encoding == 'none')

    def __repr__(self):
        if self.shape_encoding == 'none':
            shape_n = 0
        elif self.shape_encoding == 'entangled':
            shape_n = 1
        elif self.shape_encoding == 'disentangled':
            shape_n = 2

        if self.position_encoding == 'double_per_node':
            pos_n = 4
        elif self.position_encoding == 'disentangled':
            pos_n = 2
        elif self.position_encoding == 'entangled':
            pos_n = 1
        elif self.position_encoding == 'none':
            pos_n = 0

        return f"shape_{self.shape_encoding}({shape_n})__pos_{self.position_encoding}({pos_n}))"

    def preprocess(self, input) -> list:
        if isinstance(input, list):
            return input
        else:
            return [(i, p) for i, p in enumerate(input.split('_')) if p != '0']

    def register_get_item(self, item: str):
        if item not in self.attrs:
            self.attrs[item] = len(self.attrs)
        return self.attrs[item]

    def get_shape_attributes(self, input: list):
        i1, i2 = input
        i1, i2 = "shape"+str(i1[1]), "shape"+str(i2[1])
        if self.shape_encoding == 'none':
            return []
        elif self.shape_encoding == 'entangled':
            return [self.register_get_item(i1+i2)]
        elif self.shape_encoding == 'disentangled':
            return [self.register_get_item(i1), self.register_get_item(i2)]

    def get_position_attributes(self, input: list):
        i1, i2 = input
        i1, i2 = "pos"+str(i1[0]), "pos"+str(i2[0])
        if self.position_encoding == 'none':
            return []
        elif self.position_encoding == 'entangled':
            return [self.register_get_item(i1+i2)]
        elif self.position_encoding == 'disentangled':
            return [self.register_get_item(i1), self.register_get_item(i2)]
        elif self.position_encoding == 'double_per_node':
            i11, i12 = self.pos_attr_double_per_node(i1)
            i21, i22 = self.pos_attr_double_per_node(i2)
            return [self.register_get_item(i11), self.register_get_item(i12), self.register_get_item(i21), self.register_get_item(i22)]

    def process_string(self, input: Union[str, list]):
        input = self.preprocess(input)
        shape_attrs = self.get_shape_attributes(input)
        pos_attrs = self.get_position_attributes(input)

        if self.shape_encoding == 'disentangled' and self.position_encoding in ['disentangled', 'double_per_node']:
            result = []
            while shape_attrs or pos_attrs:
                result.append(shape_attrs.pop(0))
                if self.position_encoding == 'disentangled':
                    result.append(pos_attrs.pop(0))
                elif self.position_encoding == 'double_per_node':
                    result.append(pos_attrs.pop(0))
                    result.append(pos_attrs.pop(0))
        else:
            result = shape_attrs + pos_attrs

        return result

    def pos_attr_double_per_node(self, i):
        return "up_" + str(int(i[-1]) > 1), "left_" + str(int(i[-1]) % 2 == 0)


if __name__ == '__main__':
    attributizer = Attributizer('disentangled', 'double_per_node')
    data = ['bunny_rabbit_0_0', 'rabbit_rabbit_0_0', 'castle_cat_0_0']
    for d in data:
        print(attributizer.process_string(d))
