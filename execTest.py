# from pymooCFD.studies.oscillCyl_x2 import optRun
from pymooCFD.studies.rans_const_opt import exec_test as test_rans_const_opt
from pymooCFD.studies.rans_jet_opt import exec_test as test_rans_jet_opt
from pymooCFD.studies.oscill_cyl_opt import exec_test as test_oscill_cyl_opt


def main():
    # test_rans_const_opt()
    # test_rans_jet_opt()
    test_oscill_cyl_opt()

if __name__ == '__main__':
    main()
