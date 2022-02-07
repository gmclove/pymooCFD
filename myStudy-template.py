# @Author: glove
# @Date:   2021-12-14T16:04:07-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-16T10:15:10-05:00
from pymooCFD.studies.oscillCylinder import MyOptStudy, BaseCase

from pymooCFD.core.cfdCase import CFDCase
class MyBaseCase(CFDCase):
    pass

BaseCase = MyBaseCase

from pymooCFD.core.optStudy import OptStudy
