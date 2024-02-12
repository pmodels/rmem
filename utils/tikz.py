import numpy as np
import matplotlib as mpl
from enum import Enum

#===============================================================================

# from: https://spectrum.adobe.com/page/color-for-data-visualization/#Resources
# base color: #0db5af
# base color: #0db5af #4047ca #f68410 #de3d83 #7e84fb #73e06b #137cf3 #7325d3 #e8c500 #cb5c01
# from https://personal.sron.nl/~pault/
# base colors: #4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB
# variations from https://www.learnui.design/tools/data-color-picker.html#palette with grey as the neutral and 3 colors
col_pault = dict()
# blue
col_pault["blue"]=dict()
col_pault["blue"][0]="4477AA"
col_pault["blue"][1]="8598b3"
# red
col_pault["red"]=dict()
col_pault["red"][0]="EE6677"
col_pault["red"][1]="da9498"
# magenta EE3377
col_pault["magenta"]=dict()
col_pault["magenta"][0]="EE3377"
col_pault["magenta"][1]="dc8498"
# green
col_pault["green"]=dict()
col_pault["green"][0]="228833"
col_pault["green"][1]="78a377"
#teal
col_pault["teal"]=dict()
col_pault["teal"][0]="009988"
col_pault["teal"][1]="78aba1"
# yellow
col_pault["yellow"]=dict()
col_pault["yellow"][0]="CCBB44"
col_pault["yellow"][1]="c7bb83"
# cyan
col_pault["cyan"]=dict()
col_pault["cyan"][0]="66CCEE"
col_pault["cyan"][1]="99c4d4"
# purple
col_pault["purple"]=dict()
col_pault["purple"][0]="AA3377"
col_pault["purple"][1]="b77c98"
# grey
col_pault["grey"]=dict()
col_pault["grey",0]="BBBBBB"

# starting from a color code (inspiration at https://www.canva.com/colors/color-meanings/)

col_gru = dict()
# blue
col_gru["blue"]=dict()
col_gru["blue"][0]="255e7e"
col_gru["blue"][1]="73a0bc"
col_gru["blue"][2]="c0e7ff"
# magenta
col_gru["magenta"]=dict()
col_gru["magenta"][0]="e7298a"
col_gru["magenta"][1]="f68eb7"
col_gru["magenta"][2]="fcd9e5"
# yelllow
col_gru["yellow"]=dict()
col_gru["yellow"][0]="ddaa33"
col_gru["yellow"][1]="e9c586"
col_gru["yellow"][2]="ebe1d3"
# teal
col_gru["teal"]=dict()
col_gru["teal"][0]="008080"
col_gru["teal"][1]="66a09f"
col_gru["teal"][2]="a5c1c0"
# purple
col_gru["purple"]=dict()
col_gru["purple"][0]="762A83"
col_gru["purple"][1]="9970AB"
col_gru["purple"][2]="C2A5CF"

#===============================================================================

AXIS = Enum('AXIS',['loglog', 'xlog', 'ylog','xy'])
RMEM = Enum('RMEM',['p2p', 'put', 'put_trigr', 'put_fast', 'p2p_fast', 'p2p_trigr', 'put_trigr_fast', 'p2p_trigr_fast'])
LINESTYLE = Enum('LINESTYLE',['solid', 'dashed'])
MARKSTYLE = Enum('MARKSTYLE',['square', 'circle', 'diamond','triangle','pentagon'])
PLOT = Enum('PLOT',['latency', 'ratio_map', 'msg'])
TYPE = Enum('TYPE',['plot', 'reset'])

class plot:
  def __init__(self,folder,data,data_list,prov,msg,p2p_idx=0,name="",break_line=False,injection=False,bandwidth=200,case=[RMEM.p2p,RMEM.p2p_fast,RMEM.put,RMEM.put_fast],legend=[""],linestyle=None,type=TYPE.plot):
    self.folder_dir = folder
    self.data_dir = data
    self.data_list = data_list
    self.prov_list = prov
    self.msg_list = msg
    self.do_injection = injection
    if(injection):
        self.case_list = [RMEM.put_fast]
    else:
        self.case_list = case
    self.p2p_idx = p2p_idx
    self.name = name
    self.break_line = break_line
    self.bandwidth=bandwidth
    self.llist = legend
    self.linestyle = linestyle
    self.type = type
    self.max_mark = 4
    

# hexa code, no '#' in front!
# from https://personal.sron.nl/~pault/#tab:blindvision
# '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'.
# '#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499'. Bad data: '#DDDDDD'.
colors=dict()
colors[RMEM.p2p] = col_pault["teal"]
colors[RMEM.put] = col_pault["yellow"]
colors[RMEM.put_trigr] = col_pault["green"]
colors[RMEM.put_fast] = col_gru["magenta"]
colors[RMEM.p2p_fast] = col_gru["blue"]
colors[RMEM.p2p_trigr] = col_pault["cyan"]
colors[RMEM.put_trigr_fast] = col_pault["red"]
colors[RMEM.p2p_trigr_fast] = col_pault["blue"]

colgen=dict()
colgen["teal"] = col_gru["teal"]
colgen["yellow"] = col_gru["yellow"]
colgen["green"] = col_pault["green"]
colgen["magenta"] = col_gru["magenta"]
colgen["blue"] = col_gru["blue"]
colgen["cyan"] = col_pault["cyan"]
colgen["red"] = col_pault["red"]
colgen["grey"] = col_pault["grey"]
colgen["purple"] = col_pault["purple"]
colgen_idx = dict()
colgen_idx["teal"] =0
colgen_idx["yellow"] =0
colgen_idx["green"] = 0
colgen_idx["magenta"] = 0
colgen_idx["blue"] = 0
colgen_idx["cyan"] = 0
colgen_idx["red"] = 0
colgen_idx["grey"] = 0
colgen_idx["purple"] = 0

idx_list = dict()
idx_list[RMEM.p2p] = 1
idx_list[RMEM.put] = 2
idx_list[RMEM.put_trigr] = 3
idx_list[RMEM.put_fast] = 4
idx_list[RMEM.p2p_fast] = 5
idx_list[RMEM.p2p_trigr] = 6
idx_list[RMEM.put_trigr_fast] = 7
idx_list[RMEM.p2p_trigr_fast] = 8

colors_idx = dict()
colors_idx[RMEM.p2p] =0
colors_idx[RMEM.put] = 0
colors_idx[RMEM.put_trigr] = 0
colors_idx[RMEM.put_fast] = 0
colors_idx[RMEM.p2p_fast] = 0
colors_idx[RMEM.p2p_trigr] =0
colors_idx[RMEM.put_trigr_fast] =0
colors_idx[RMEM.p2p_trigr_fast] =0

_linewidth = 0.95

mark_idx = int()
mark_list=dict()
mark_list[0] = MARKSTYLE.circle
mark_list[1] = MARKSTYLE.square
mark_list[2] = MARKSTYLE.diamond
mark_list[3] = MARKSTYLE.triangle
mark_list[4] = MARKSTYLE.pentagon
mark_list[5] = MARKSTYLE.circle
mark_list[6] = MARKSTYLE.circle

def leg_2_colkey(leg):
    if leg.find("p2p")>=0:
        return "blue"
    elif leg.find("dc")>=0:
        return "yellow"
    elif leg.find("cq data")>=0:
        return "purple"
    elif leg.find("rcntr")>=0:
        return "magenta"
    elif leg.find("fence")>=0:
        return "green"
    elif leg.find("order")>=0:
        return "green"
    else:
        print(f"col 2 leg failed with {leg}")
        return "grey"

def rmem_2_legend(case):
    print(f"looking for a legend for {case}")
    if(case == RMEM.p2p):
        return "p2p"
    elif(case == RMEM.put):
        return "put"
    elif(case == RMEM.put_trigr):
        return "put trigr"
    elif(case == RMEM.put_fast):
        return "put fast"
    elif(case == RMEM.p2p_fast):
        return "p2p prepost"
    elif(case == RMEM.p2p_trigr):
        return "p2p trigger"
    elif(case == RMEM.put_trigr_fast):
        return "put trigger fast"
    elif(case == RMEM.p2p_trigr_fast):
        return "p2p trigger fast"
    
def rmem_2_linestyle(case):
    if (case == RMEM.p2p_fast or case == RMEM.put_fast):
        return LINESTYLE.solid
        # return LINESTYLE.dashed
    else:
        return LINESTYLE.solid

def axis_2_tikz(axis):
    if (axis == AXIS.xy):
        return "axis"
    elif (axis == AXIS.loglog):
        return  "loglogaxis"
    elif (axis == AXIS.xlog):
            return  "xlogaxis"
    elif (axis == AXIS.ylog):
            return  "ylogaxis"

def linestyle_2_tikz(line):
    if (line == LINESTYLE.solid):
        return "solid"
    elif (line == LINESTYLE.dashed):
        return "dashed"

# ['square', 'circle', 'diamond','triangle','pentagon'])
def markstyle_2_tikz(mark):
    if (mark == MARKSTYLE.square):
        return "mark=square*"
    elif (mark == MARKSTYLE.circle):
        return "mark=*"
    elif (mark == MARKSTYLE.diamond):
        return "mark=diamond*"
    elif (mark == MARKSTYLE.triangle):
        return "mark=triangle*"
    elif (mark == MARKSTYLE.pentagon):
        return "mark=pentagon*"
    else:
        return "mark=o"

    
#===============================================================================
def tikz_plot(filename,plot_list,header="",standalone=True,kind=PLOT.latency):
    # reset color counters
    for key in colors_idx:
        colors_idx[key]=0
    for key in colgen_idx:
        colgen_idx[key]=0
    # reset mark counter
    global mark_idx
    mark_idx = 0

    # default tikz opts
    tikz_opt = f"{header}"
                
    if(kind is PLOT.latency):
        tikz_opt = tikz_opt + "," + \
            "xlabel={msg size [B]},\n"\
            "ylabel={time/msg [$\mu$s]},\n"\
            "major grid style={thick},ymajorgrids=true,yminorgrids=true,\n"\
            "xtick={4,32,256,2048,16384,262144,4194304},\n"\
            "xticklabels={4,32,256,2ki,16ki,262ki,4Mi},\n"\
    
    elif(kind is PLOT.msg):
        tikz_opt = tikz_opt + "," + \
            "xlabel={number of msgs},"\
            "ylabel={time/msg [$\mu$s]},"\
            "xtick={1,4,16,64,256,1024},"\
            "xticklabels={1,4,16,64,256,1024},"\
            "legend style={anchor=north east,at={(0.99,0.99)},},"\
            "log ticks with fixed point,"\
            "major grid style={thick},ymajorgrids=true,yminorgrids=true,\n"\
            # "ymin=0.2,"\
            
    elif(kind is PLOT.ratio_map):
        tikz_opt = tikz_opt + "," + \
            "xlabel={msg size [B]},"\
            "ylabel={msg },"\
            "xtick={4,32,256,2048,16384,262144,4194304},"\
            "xticklabels={4,32,256,2ki,16ki,262ki,4Mi},"\
            "major grid style={thick},ymajorgrids=true,yminorgrids=true,\n"\
            
    #---------------------------------------------------------------------------
    # open the file and write the header if needed
    file = open(filename,"w+")
    print(f"\n>>writing {filename}<<")
    if(standalone):
        tikz_header(file,kind=kind)
    else:
        # search for the first title in the plot list
        title = ""
        for pplot in plot_list:
            if(len(pplot.name)):
                title=pplot.name
                break
        tikz_axis_open(file,title,tikz_opt=tikz_opt,kind=kind)
    #---------------------------------------------------------------------------
    # let's gooooooo
    plot_idx = 0
    raw=dict()
    for plot in plot_list:
        print(f"plot type = {plot.type}")
        if(plot.type is TYPE.reset):
            print(f"--- RESET COUNTERS ---")
            # reset color counters
            for key in colors_idx:
                colors_idx[key]=0
            for key in colgen_idx:
                colgen_idx[key]=0
            # reset mark counter
            mark_idx = 0
            # 
            continue
        leg_idx = 0
        #-----------------------------------------------------------------------
        # if standalone, create a new plot
        if(standalone):
            title=plot.name
            tikz_axis_open(file,f"{title}",tikz_opt=tikz_opt,kind=kind)
        #-----------------------------------------------------------------------
        if(kind is PLOT.latency or kind is PLOT.msg):
            print(f"doing plot with kind = {kind}")
            tikz_plot_latency_msg(plot,file,leg_idx,kind=kind)
        elif(kind is PLOT.ratio_map):
            tikz_plot_ratio(plot,file)
        #-----------------------------------------------------------------------
        # if standalone, need to close the plot, if not keep going
        if(standalone):
            tikz_axis_close(file,break_line=plot.break_line,kind=kind)
        plot_idx = plot_idx +1
    #---------------------------------------------------------------------------
    # close it
    if(standalone):
        tikz_footer(file)
    else:
        tikz_axis_close(file,break_line=plot.break_line,kind=kind)
    file.close()


def tikz_plot_ratio(plot,tikz_file,time_ref=RMEM.p2p):
    for prov in plot.prov_list:
        for case in plot.case_list:
            min_val = 1000
            max_val = 0
            # write the start of the tikz add plot
            tikz_file.write("\\addplot [matrix plot*,point meta=explicit,\n"
                            "] coordinates {\n")
            for msg in plot.msg_list:
                for data in plot.data_list:
                    data_folder = f"{plot.folder_dir}/{plot.data_dir}/data_{data}"
                    data_file = f"{data_folder}/r1_msg{msg}_{prov}.txt"
                    data = np.loadtxt(data_file,delimiter=",")
                    size_data = data[:,0]
                    time_p2p = data[:,idx_list[time_ref]]
                    time_data = data[:,idx_list[case]]
                    text = ""
                    for i,size in enumerate(size_data[0:-1]):
                        if(size<= 1<<18):
                            # ratio = (time_data[i]-time_p2p[i])/time_p2p[i]
                            ratio = time_p2p[i]/(time_data[i])
                            text = text + f"({size},{msg}) [{ratio}] "
                            if(ratio < min_val):
                                min_val = ratio
                            if(ratio > max_val):
                                max_val = ratio
                    tikz_file.write(f"{text}\n\n")
            tikz_file.write("};\n")
            tikz_file.write("\\pgfplotsset{\n")
            if(min_val < 1.0):
                white_value = int((1.0-min_val)/(max_val-min_val)*1000)
            else:
                white_value = 0
            # tikz_file.write(f"colormap={{CM}}{{rgb255(0)=(33,102,172) rgb({white_value})=(1,1,1) rgb255(1000)=(178,24,43)}},\n")
            if(white_value > 25):
                tikz_file.write(f"colormap={{CM}}{{rgb255(0)=(0,119,187) rgb({white_value})=(1,1,1) rgb255(1000)=(204,51,17)}},\n")
            else:
                tikz_file.write(f"colormap={{CM}}{{rgb(0)=(1,1,1) rgb255(1000)=(204,51,17)}},\n")
            tikz_file.write("colorbar sampled,\n")
            tikz_file.write("grid=none,\n")
            tikz_file.write("colorbar style={width=0.25cm},")
            # tikz_file.write("colorbar style={title={$\dfrac{T_{R}-T{s}}{T_s}$},},")
            tikz_file.write("colorbar style={title={$T_{p2p}/T_{put}$},},\n")
            # tikz_file.write(f"point meta min={min(min_val,1)},\n")
            # tikz_file.write(f"point meta max={max(1,max_val)},\n")
            tikz_file.write("};\n")

def tikz_plot_latency_msg(plot,file,leg_idx,kind=PLOT.latency):
    global mark_idx
    global mark_list
    print(f"mark_idx is now {mark_idx}")
    for prov in plot.prov_list:
            for msg in plot.msg_list:
                for case in plot.case_list:
                    #-----------------------------------------------------------
                    # P2P
                    #-----------------------------------------------------------
                    if(case == RMEM.p2p or case==RMEM.p2p_fast) and (plot.p2p_idx >=0):
                        if kind is PLOT.latency:
                            data_file = f"{plot.folder_dir}/{plot.data_dir}/data_{plot.p2p_idx}/r1_msg{msg}_{prov}.txt"
                        elif kind is PLOT.msg:
                            data_file = f"{plot.folder_dir}/{plot.data_dir}/data_{plot.p2p_idx}/r1_size{msg}_{prov}.txt"
                        print(f"doing plot with kind = {kind}, data file = {data_file}")
                        # get the legend either from the legend list of from the automatic conversion
                        if(leg_idx < len(plot.llist)):
                            legend = plot.llist[leg_idx]
                        else:
                            legend = rmem_2_legend(case)
                        # get the color
                        col_key = leg_2_colkey(f"{rmem_2_legend(case)}")
                        color = colgen[col_key][colgen_idx[col_key]]
                        colgen_idx[col_key]=colgen_idx[col_key]+1
                        if plot.linestyle:
                            linestyle = plot.linestyle
                        else:
                            linestyle = rmem_2_linestyle(case)
                        markstyle = mark_list[mark_idx]
                        print(f"markstyle = {markstyle}")
                        # loop over the mark_idx
                        mark_idx = (mark_idx + 1)%plot.max_mark
                        print(f"doing case {case} from data_{plot.p2p_idx}: {legend} from {data_file}")
                        tikz_addplot(file,0,idx_list[case],data_file,legend=legend,do_injection=plot.do_injection,linestyle=linestyle,color=color, marker=markstyle)
                        leg_idx = leg_idx +1;
                    #-----------------------------------------------------------
                    # RMA
                    #-----------------------------------------------------------
                    else:
                        for data in plot.data_list:
                            data_folder = f"{plot.folder_dir}/{plot.data_dir}/data_{data}"
                            # data_file = f"{data_folder}/r1_msg{msg}_{prov}.txt"
                            if kind is PLOT.latency:
                                data_file = f"{data_folder}/r1_msg{msg}_{prov}.txt"
                            elif kind is PLOT.msg:
                                data_file = f"{data_folder}/r1_size{msg}_{prov}.txt"
                            print(f"doing plot with kind = {kind}, data file = {data_file}")
                            rma = tikz_read_info(data_folder)
                            case_name = rmem_2_legend(case)
                            # get the legend
                            if(leg_idx < len(plot.llist)):
                                legend = plot.llist[leg_idx]
                            else:
                                legend = f"{rmem_2_legend(case)} - {rma}"
                            # skip if cannot do it
                            # if (case_name.find("put fast")>=0 and rma.find("dc")>=0) or (case_name.find("put fast")>=0 and rma.find("fence")>=0):
                            #     continue
                            # color = colors[case][colors_idx[case]]
                            # get the color + linestyle
                            col_lookingfor = f"{rmem_2_legend(case)} - {rma}"
                            print(f"looking for color for {col_lookingfor}")
                            col_key = leg_2_colkey(f"{rmem_2_legend(case)} - {rma}")
                            color = colgen[col_key][colgen_idx[col_key]]
                            colgen_idx[col_key]=colgen_idx[col_key]+1
                            if plot.linestyle:
                                linestyle = plot.linestyle
                            else:
                                linestyle = rmem_2_linestyle(case)
                            markstyle = mark_list[mark_idx]
                            print(f"markstyle = {markstyle}")
                            mark_idx = (mark_idx + 1)%plot.max_mark
                            # tikz_addplot(file,0,idx_list[case],data_file,legend=legend,do_injection=plot.do_injection,linestyle=linestyle,color=color)
                            print(f"doing case {case} from data_{data}: {legend} from {data_file}")
                            tikz_addplot(file,0,idx_list[case],data_file,legend=legend,do_injection=plot.do_injection,linestyle=linestyle,color=color, marker=markstyle)
                            leg_idx = leg_idx +1;
        



#===============================================================================

_color_id = int(0)

def tikz_addplot(tikz_file,x_idx,y_idx, datafile,
            do_injection=False,
            legend="",
            linestyle=LINESTYLE.solid,
            marker=MARKSTYLE.circle,
            color=[0,0,0]):
    global _color_id
    ncolors=len(mpl.cm.get_cmap('tab10').colors)    
    #---------------------------------------------------------------------------
    # if(bool(color)):
    if(color == [0,0,0]):
        color = mpl.cm.get_cmap('tab10').colors[_color_id]
        tikz_file.write(f"\\definecolor{{color_{_color_id}}}{{RGB}}{{{color[0]*255.0},{color[1]*255.0},{color[2]*255.0}}}\n")
    else:
        tikz_file.write(f"\\definecolor{{color_{_color_id}}}{{HTML}}{{{color}}}\n")
    # print(f"color_id = {_color_id} -> color = %{color}")
    color_string = f"color_{_color_id}"
    _color_id = (_color_id + 1)%ncolors
    # else:
    #     color_string = ""
    tikz_line = linestyle_2_tikz(linestyle)
    tikz_mark = markstyle_2_tikz(marker)
    #---------------------------------------------------------------------------
    # do it!
    if(do_injection):
        dataf = datafile.replace("r1","r0")
        datafile = dataf
    tikz_file.write(f"\\addplot+[{tikz_line},{color_string},{tikz_mark},line width={_linewidth}pt, mark size={1.5*_linewidth}pt, mark options={{fill=none}}] table[col sep=comma, x index={x_idx}, y index={y_idx}] {{{datafile}}};\n")
    if(bool(legend)):
        tikz_file.write(f"\\addlegendentryexpanded{{{legend}}};\n")
    #     # tikz_line = linestyle_2_tikz(LINESTYLE.dashed)
    #     tikz_file.write(f"\\addplot+[{tikz_line},{color_string},forget plot,line width={_linewidth}pt] table[col sep=comma, x index={x_idx}, y index={y_idx}] {{{dataf}}};\n")
        
    

def tikz_header(tikz_file,tikz_opt="",kind=PLOT.latency):
    tikz_file.write("\\documentclass[border=5mm,varwidth=\\maxdimen]{standalone}\n")
    tikz_file.write("\\usepackage[dvipsnames]{xcolor}\n")
    tikz_file.write("\\usepackage{pgfplots}\n")
    tikz_file.write("\\usepackage{pgfmath,pgffor}\n")
    tikz_file.write("\\pgfplotsset{compat=1.15}\n")
    tikz_file.write("%------------------------------------------------------------------------------------------------\n")
    tikz_file.write("\\def\\picfont{\\sffamily}\n")
    tikz_file.write("\\tikzset{font={\\picfont \\small}}\n")
    tikz_file.write("\n")
    tikz_file.write("\\definecolor{lightgray204}{RGB}{204,204,204}\n")
    tikz_file.write("%------------------------------------------------------------------------------------------------\n")
    tikz_file.write("\\pgfplotscreateplotcyclelist{colorlist}{\n")
    tikz_file.write("    {MidnightBlue},\n")
    tikz_file.write("    {BurntOrange},\n")
    tikz_file.write("    {OliveGreen},\n")
    tikz_file.write("    {BrickRed},\n")
    tikz_file.write("    {Periwinkle},\n")
    tikz_file.write("    {CadetBlue}% <-- don't add a comma here\n")
    tikz_file.write("}\n")
    tikz_file.write("\\pgfplotscreateplotcyclelist{marklist}{\n")
    tikz_file.write("    {mark=o},\n")
    tikz_file.write("    {mark=x},\n")
    tikz_file.write("    {mark=square},\n")
    tikz_file.write("    {mark=triangle},\n")
    tikz_file.write("    {mark=diamond}% <-- don't add a comma here\n")
    tikz_file.write("}\n")
    tikz_file.write("%------------------------------------------------------------------------------------------------\n")
    tikz_file.write("\\pgfplotsset{every axis/.append style={\n")
    tikz_file.write("log basis x=2,\n")
    if(kind is PLOT.latency or kind is PLOT.msg):
        tikz_file.write("% grid on\n")
        tikz_file.write("ymajorgrids=true,\n")
        tikz_file.write("xmajorgrids=true,\n")
        tikz_file.write("% grid\n")
        tikz_file.write("grid=both,\n")
        tikz_file.write("%minor tick num=2,\n")
        tikz_file.write("%every minor tick/.style={minor tick length=0pt,dashed},\n")
        tikz_file.write("% line width\n")
        tikz_file.write("% position the legend\n")
        tikz_file.write("log ticks with fixed point,\n")
        tikz_file.write("legend pos=north west,\n")
        # tikz_file.write("legend style={fill opacity=0.8, draw opacity=1, text opacity=1, draw=lightgray204,font={\\picfont\\tiny}},\n")
        tikz_file.write("legend cell align={left},\n")
        tikz_file.write("mark size = 0.5pt,\n")
        tikz_file.write("cycle list name = colorlist,\n")
        tikz_file.write("cycle multi list = {\n")
        tikz_file.write("	marklist\\nextlist\n")
        tikz_file.write("	colorlist\\nextlist},\n")
    if(kind is PLOT.ratio_map):
        tikz_file.write("colormap={CM}{rgb255=(33,102,172) rgb=(1,1,1) rgb255=(178,24,43)},\n")
        tikz_file.write("colorbar sampled,\n")
        # tikz_file.write("colorbar style={title={$\dfrac{T_{R}-T{s}}{T_s}$},},")
        tikz_file.write("colorbar style={title={$T_{p2p}/T_{put}$},},\n")
        tikz_file.write("point meta min=0,\n")
        tikz_file.write("point meta max=2,\n")
    # elif(kind is PLOT.ratio_map):
    #     tikz_file.write("grid=off,\n")
    tikz_file.write("},\n")
    # tikz_file.write("every axis plot/.append style={line width=0.5pt}\n")
    if(bool(tikz_opt)):
        tikz_file.write(f"{tikz_opt},\n")
    tikz_file.write("}\n")
    tikz_file.write("\n")
    tikz_file.write("\\begin{document}\n")

def tikz_footer(tikz_file):
    tikz_file.write("\\end{document}\n")



def tikz_axis_open(tikz_file,title,tikz_opt="",kind=PLOT.latency):
    opt = tikz_opt
    if(kind is PLOT.latency or kind is PLOT.msg):
        axis=AXIS.loglog
    else:
        axis=AXIS.loglog
        opt = opt + "enlargelimits=false, axis on top,colorbar,"
    global _color_id
    _color_id = 0
    tikz_file.write("\\begin{tikzpicture}\n\pgfplotsset{compat=1.18}\n")
    if(len(title)):
        opt = opt + f"title={{{title}}},\n"
    tikz_file.write(f"\\begin{{{axis_2_tikz(axis)}}}[{opt}]\n")

def tikz_axis_close(tikz_file,break_line=False,bw=200,kind=PLOT.latency):
    if(kind is PLOT.latency):
        axis=AXIS.loglog
        # plot the bandwidth ,forget plot
        x1 = 1<<17
        x2 = 1<<23
        y1 = x1/(bw*1e9)*8e+6
        y2 = x2/(bw*1e9)*8e+6
        tikz_file.write(f"\\addplot[mark=none, black,line width={_linewidth}pt,dashed,forget plot] coordinates {{({x1},{y1}) ({x2},{y2})}};\n")
    else:
        axis=AXIS.loglog
    
    tikz_file.write(f"\\end{{{axis_2_tikz(axis)}}}\n")
    tikz_file.write("\\end{tikzpicture}\n")
    if(break_line):
        tikz_file.write("\\begin{center}\n")
        tikz_file.write("\\noindent\\rule{0.5\\paperwidth}{2pt}\n")
        tikz_file.write("\\end{center}\n")

def tikz_read_info(folder):
    key = ""
    file = open(f"{folder}/rmem.info","r")
    for line in file:
        if line.find("ready-to-receive")>=0:
            if line.find("TAGGED")>=0:
                key = key + "tag"
            elif line.find("ATOMIC")>=0:
                key = key + "atom"
            elif line.find("MSG")>=0:
                key = key + "am"
            else:
                AssertionError("unknown")
        if line.find("remote completion")>=0:
            if line.find("CQ_DATA")>=0:
                key = key + " - cq data"
            elif line.find("DELIVERY COMPLETE")>=0:
                key = key + " - dc"
            elif line.find("REMOTE COUNTER")>=0:
                key = key + " - rcntr"
            elif line.find("FENCE")>=0:
                key = key + " - fence"
            elif line.find("ORDER")>=0:
                key = key + " - order"
            else:
                AssertionError("unknown")
    file.close()
    #end of the line
    # file = open(f"{folder}/rmem.info","r")
    # for line in file:
    #     if line.find("CUDA GPU")>=0:
    #         key = key + " - CUDA"
    #     elif line.find("HIP GPU")>=0:
    #         key = key + " - HIP"
    #     elif line.find("NO GPU")>=0:
    #         key = key + " - CPU"
    #     else:    
    #         AssertionError("unknown")
    # file.close()
    return key