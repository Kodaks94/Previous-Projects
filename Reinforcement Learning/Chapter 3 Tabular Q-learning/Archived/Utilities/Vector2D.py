

class Vector2D(object):


    def __init__(self, x,y):

        self.x = float(x)
        self.y = float(y)

    def get(self):
        return [self.x, self.y]
    def set(self, vector):
        self.x = vector.x
        self.y = vector.y
    def convert_to_coords(self, units):
        unit = units / 2
        row_coords = self.y * units + unit
        column_coords = self.x * units + unit
        return [column_coords, row_coords, column_coords + 30, row_coords + 30]
    def get_center_coords(self, units):
        number_ = self.convert_to_numbers(units)
        unit = units /2
        x_coords = number_[0] * units + unit
        y_coords = number_[1] * units + unit
        return [x_coords, y_coords]
    def convert_to_numbers(self, units):
        #unit = units / 2
        row_number = (self.y ) / units
        column_number = (self.x ) / units
        return [int(column_number), int(row_number)]

    def convert_to_numbers_raw(state,units):
        # unit = units / 2

        x,y, e,h = state
        row_number = (y) / units
        column_number = (x) / units
        return [int(column_number), int(row_number)]

