import numpy as np
import os

path = os.path.join(os.getcwd(), "default_merton/")
dd = [10, 50, 100, 500, 1000, 5000, 10000]
Mnmax = 5
runs = 10

sol_mlp = np.zeros([len(dd), Mnmax, runs], dtype = np.float32)
tms_mlp = np.zeros([len(dd), Mnmax, runs], dtype = np.float32)
fev_mlp = np.zeros([len(dd), Mnmax, runs], dtype = np.float32)
for i in range(len(dd)):
    sol_mlp[i] = np.loadtxt(path + "mlp_sol_" + str(dd[i]) + ".csv")
    tms_mlp[i] = np.loadtxt(path + "mlp_tms_" + str(dd[i]) + ".csv")
    fev_mlp[i] = np.loadtxt(path + "mlp_fev_" + str(dd[i]) + ".csv")

# Table
txt = ""
for i in range(len(dd)):
    txt = txt + str(dd[i]) + " & Avg. Sol. & "
    for m in range(Mnmax):
        txt = txt + "$" + '{:.4f}'.format(np.nanmean(sol_mlp[i, m])) + "$"
        if m < Mnmax-1:
            txt = txt + " & "
        else:
            txt = txt + " \\\ \n"
            
    txt = txt + " & \\textit{Std. Dev.} & "
    for m in range(Mnmax):
        txt = txt + "\\textit{" + '{:.4f}'.format(np.nanstd(sol_mlp[i, m])) + "}"
        if m < Mnmax-1:
            txt = txt + " & "
        else:
            txt = txt + " \\\ \n"
            
    txt = txt + " & Avg. Eval. & "
    for m in range(Mnmax):
        t = np.nanmean(fev_mlp[i, m])
        e = np.floor(np.log10(t))
        r = t/np.power(10, e)
        if t == 0:
            txt = txt + " "
        else:
            txt = txt + "$" + '{:.2f}'.format(r) + " \\cdot 10^{" + '{:.0f}'.format(e) + "}$"
            
        if m < Mnmax-1:
            txt = txt + " & "
        else:
            txt = txt + " \\\ \n"
            
    txt = txt + " & \\textit{Avg. Time} & "
    for m in range(Mnmax):
        t = np.nanmean(tms_mlp[i, m])
        if t < 0.0001:
            txt = txt + "\\textit{<0.0001}"
        else:
            txt = txt + "\\textit{" + '{:.4f}'.format(t) + "}"
            
        if m < Mnmax-1:
            txt = txt + " & "
        else:
            txt = txt + " \\\ \n\hline \n"
        
text_file = open(path + "table.txt", "w")
n = text_file.write(txt)
text_file.close()