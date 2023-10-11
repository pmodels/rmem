from tikz import *

fig_dir = "/Users/tgillis/git-code/rmem/utils"

#===============================================================================
data_dir = "/Users/tgillis/git-research/2023_10_icl/results/"
#===============================================================================
plot_list = []
plot_list.append(plot(data_dir,data="rmem_2023-10-11-0024-576f_4732933",data_list=[0,2,1],prov=["cxi"],msg=[1], p2p_idx=0,name="CXI - CPU - LUMI",break_line=True))
plot_list.append(plot(data_dir,data="rmem_2023-10-09-2129-9d53_16850145",data_list=[0,2,1],prov=["cxi"],msg=[1], p2p_idx=0,name="CXI - CPU - PERLM"))
plot_list.append(plot(data_dir,data="rmem_2023-10-09-2129-9d53_16850145",data_list=[3,5,4],prov=["cxi"],msg=[1], p2p_idx=3,name="CXI - GPU - PERLM",break_line=True))
plot_list.append(plot(data_dir,data="benchme_2023-10-05-1859-91c3_474443",data_list=[0,2,1],prov=["verbs;ofi_rxm"],msg=[1], p2p_idx=0,name="VERBS - CPU - MLX"))
plot_list.append(plot(data_dir,data="benchme_2023-10-05-1859-91c3_474443",data_list=[3,5,4],prov=["verbs;ofi_rxm"],msg=[1], p2p_idx=3,name="VERBS - GPU - MLX",break_line=True))
# prov_list = ["cxi"]
# msg_list = [1]
# case_list = [RMEM.p2p,RMEM.put,RMEM.put_trigr]
# # data_list = [0,1,2]
# data_list = [0,1,2]
# do_injection=False

tikz_opt = "xlabel={total msg size},\
            ylabel={time [$\mu$s]},\
            xtick={4,32,256,2048,16384,262144,4194304},\
            xticklabels={4,32,256,2ki,16ki,262ki,4Mi},\
            ymin=1,\
            ymax=1000,\
            "

#===============================================================================
filename = f"{fig_dir}/test.tex"
file = open(filename,"w+")
print(f"writing {filename}")
tikz_header(file)
# Latency

plot_idx = 0
for plot in plot_list:
    tikz_axis_open(file,f"{plot.name}",AXIS.loglog,tikz_opt=tikz_opt)
    for prov in plot.prov_list:
        for msg in plot.msg_list:
            for case in plot.case_list:
                if(case == RMEM.p2p):
                    data_file = f"{plot.folder_dir}/{plot.data_dir}/data_{plot.p2p_idx}/r1_msg{msg}_{prov}.txt"
                    legend = "p2p"
                    tikz_addplot(file,0,idx_list[case],data_file,legend=legend,do_injection=plot.do_injection)
                else:
                    for data in plot.data_list:
                        print(f"doing case {case} from data_{data}")
                        data_folder = f"{plot.folder_dir}/{plot.data_dir}/data_{data}"
                        data_file = f"{data_folder}/r1_msg{msg}_{prov}.txt"
                        rma = tikz_read_info(data_folder)
                        case_name = rmem_2_legend(case)
                        legend = f"{rmem_2_legend(case)} - {rma}"
                        # if (case_name.find("fast")>=0 and rma.find("dc")>=0):
                        #     continue
                        tikz_addplot(file,0,idx_list[case],data_file,legend=legend,do_injection=plot.do_injection)

    tikz_axis_close(file,AXIS.loglog,break_line=plot.break_line)
    plot_idx = plot_idx +1
tikz_footer(file)


file.close()