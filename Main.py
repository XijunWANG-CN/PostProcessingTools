import DataAnalysis as DA
import numpy as np

filename = "data.txt"
disp_list1 = list(np.loadtxt(filename, delimiter='	', usecols=(0,), skiprows=0, dtype=float))
force_list1 = list(np.loadtxt(filename, delimiter='	', usecols=(1,), skiprows=0, dtype=float))
data1 = DA.Data(disp_list1, force_list1, 'cyc_full')
# data1.obtain_protocol(plotopt=True)
# data1.reorgdata(plotopt=True)
# data1.backbone(plotopt=True)
# data1.energy(cumulative=True, plotopt=True)
data1.yield_point("yk", plotopt=True)
# data1.plotcurve()
