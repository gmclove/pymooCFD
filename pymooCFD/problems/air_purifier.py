from pymooCFD.core.cfdCase import YALES2Case


class AirPurifier(YALES2Case):
    n_var = 2
    var_labels = ['x-location', 'y-location']
    car_type = ['real', 'real']
    xl = []
    xu = []

    n_obj = 2
    obj_labels = ['ACH', ]
