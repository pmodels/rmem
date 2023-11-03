from tikz import *

fig_dir = "/Users/tgillis/git-code/rmem/utils"

#===============================================================================
data_dir = "/Users/tgillis/git-research/2023_10_icl/results/"
dwnld_dir= "/Users/tgillis/Downloads"
#===============================================================================
nmsg = 1
plot_list = []
# data = "rmem_2023-10-16-2037-6e6d_17111443" #hybrid MR
# plot_list.append(plot(dwnld_dir,data=data,data_list=[0,1,2],prov=["cxi"],msg=[nmsg], p2p_idx=0,name=f"CXI - CPU - PERLM - {nmsg} msgs"))
# plot_list.append(plot(dwnld_dir,data=data,data_list=[3,4,5],prov=["cxi"],msg=[nmsg], p2p_idx=3,name=f"CXI - GPU - PERLM - {nmsg} msgs",break_line=True))
# plot_list.append(plot(data_dir,data="rmem_2023-10-09-2129-9d53_16850145",data_list=[0,2,1],prov=["cxi"],msg=[nmsg], p2p_idx=0,name="CXI - CPU - PERLM"))
# plot_list.append(plot(data_dir,data="rmem_2023-10-09-2129-9d53_16850145",data_list=[3,5,4],prov=["cxi"],msg=[nmsg], p2p_idx=3,name="CXI - GPU - PERLM",break_line=True))
# plot_list.append(plot(data_dir,data=data,data_list=[0,2,1],prov=["verbs;ofi_rxm"],msg=[nmsg], p2p_idx=0,name="VERBS - CPU - MLX",injection=True))
# plot_list.append(plot(data_dir,data=data,data_list=[3,5,4],prov=["verbs;ofi_rxm"],msg=[nmsg], p2p_idx=3,name="VERBS - GPU - MLX",injection=True,break_line=True))
# data="rmem_2023-10-13-0125-2f3f_4750585"
# data="rmem_2023-10-16-2330-dfcc_4779451"
# plot_list.append(plot(dwnld_dir,data=data,data_list=[0,1,2],prov=["cxi"],msg=[nmsg], p2p_idx=0,name="CXI - CPU - LUMI"))
# plot_list.append(plot(dwnld_dir,data=data,data_list=[3,4,5],prov=["cxi"],msg=[nmsg], p2p_idx=0,name="CXI - GPU - LUMI",break_line=True))
# data="rmem_2023-10-17-0203-49ab_4780323"
# plot_list.append(plot(dwnld_dir,data=data,data_list=[0,1,2],prov=["cxi"],msg=[nmsg], p2p_idx=0,name="CXI - CPU - LUMI"))
# plot_list.append(plot(dwnld_dir,data=data,data_list=[3,4,5],prov=["cxi"],msg=[nmsg], p2p_idx=3,name="CXI - GPU - LUMI",break_line=True))
# data="rmem_2023-10-17-2253-e1f9_4791530"
# plot_list.append(plot(dwnld_dir,data=data,data_list=[0,1,2],prov=["cxi"],msg=[nmsg], p2p_idx=0,name=f"CXI - CPU - LUMI - {nmsg} msgs"))
# # plot_list.append(plot(dwnld_dir,data=data,data_list=[5,6,7,8,9],prov=["cxi"],msg=[nmsg], p2p_idx=5,name=f"CXI - GPU - LUMI - {nmsg} msgs",break_line=True))

# data="benchme_2023-10-20-0449-4b23_487885"# gdr comparison
# plot_list.append(plot(dwnld_dir,data=data,data_list=[0,1],prov=["verbs;ofi_rxm"],msg=[nmsg], p2p_idx=-1,name=f"VERBS:RXM - GPU - {nmsg} msgs",case=[RMEM.p2p]))
data="benchme_2023-10-27-1845-43ed_496494"
plot_list.append(plot(dwnld_dir,data=data,data_list=[0,1,2],prov=["verbs;ofi_rxm"],msg=[nmsg], p2p_idx=0,name=f"VERBS - CPU - MLX - {nmsg} msgs"))
plot_list.append(plot(dwnld_dir,data=data,data_list=[3,4,5],prov=["verbs;ofi_rxm"],msg=[nmsg], p2p_idx=3,name=f"VERBS - GPU - MLX - {nmsg} msgs",break_line=True))
plot_list.append(plot(dwnld_dir,data=data,data_list=[0],prov=["verbs;ofi_rxm"],msg=[nmsg], p2p_idx=-1,name=f"VERBS - CPU - MLX - {nmsg} msgs",case=[RMEM.p2p,RMEM.p2p_trigr,RMEM.put,RMEM.put_fast,RMEM.put_trigr,RMEM.put_trigr_fast]))
plot_list.append(plot(dwnld_dir,data=data,data_list=[3],prov=["verbs;ofi_rxm"],msg=[nmsg], p2p_idx=-1,name=f"VERBS - GPU - MLX - {nmsg} msgs",case=[RMEM.p2p,RMEM.p2p_trigr,RMEM.put,RMEM.put_fast,RMEM.put_trigr,RMEM.put_trigr_fast],break_line=True))

data="rmem_2023-10-27-1200-079e_17493129"
# data="rmem_2023-10-27-1319-1cf6_17494426" # custom rdv sizes 1024 and 1024
data="rmem_2023-10-27-1518-b683_17499960" # 1024, 1024, no inline
plot_list.append(plot(dwnld_dir,data=data,data_list=[0,1,2],prov=["cxi"],msg=[nmsg], p2p_idx=0,name=f"CXI - CPU - PERLM - {nmsg} msgs"))
plot_list.append(plot(dwnld_dir,data=data,data_list=[3,4,5],prov=["cxi"],msg=[nmsg], p2p_idx=3,name=f"CXI - GPU - PERLM - {nmsg} msgs",case=[RMEM.p2p,RMEM.put_fast],break_line=True))
plot_list.append(plot(dwnld_dir,data=data,data_list=[1],prov=["cxi"],msg=[nmsg], p2p_idx=-1,name=f"CXI - CPU - PERLM - {nmsg} msgs",case=[RMEM.p2p,RMEM.p2p_trigr,RMEM.put,RMEM.put_fast,RMEM.put_trigr,RMEM.put_trigr_fast]))
plot_list.append(plot(dwnld_dir,data=data,data_list=[4],prov=["cxi"],msg=[nmsg], p2p_idx=-1,name=f"CXI - GPU - PERLM - {nmsg} msgs",case=[RMEM.p2p,RMEM.p2p_trigr,RMEM.put,RMEM.put_fast,RMEM.put_trigr,RMEM.put_trigr_fast],break_line=True))


#===============================================================================
filename = f"{fig_dir}/test.tex"
tikz_plot(filename,plot_list,header="ymin=1,")