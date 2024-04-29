class Input:
    def __init__(self):
        self.inputs = {}
        self.current_length = 0

    def add_input(self, name, length):
        self.inputs[name] = (self.current_length, self.current_length + length)
        self.current_length += length
        return self

    def get_input(self, input, name):
        a, b = self.inputs[name]
        return input[a:b]

    def set_input(self, input, name, value):
        a, b = self.inputs[name]
        input[a:b] = value
        return input

    def size_of(self, name):
        a, b = self.inputs[name]
        return b - a

    def bounds_of(self, name):
        a, b = self.inputs[name]
        return a, b

    def get_size(self):
        return self.current_length - 1

