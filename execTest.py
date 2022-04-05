from pymooCFD.studies.oscillCyl_x2 import optStudy

def main():
    # optStudy.runTestCase()
    optStudy.testCase.meshStudy.run()
    optStudy.run()


if __name__=='__main__':
    main()
