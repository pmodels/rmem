import numpy as np
import matplotlib as mpl
from enum import Enum



class plot:
  def __init__(self,folder,data,data_list,prov,msg,p2p_idx=0,name="time",break_line=False):
    self.folder_dir = folder
    self.data_dir = data
    self.data_list = data_list
    self.prov_list = prov
    self.msg_list = msg
    self.case_list = [RMEM.p2p,RMEM.put,RMEM.put_trigr]
    self.do_injection = False
    self.p2p_idx = p2p_idx
    self.name = name
    self.break_line = break_line
 

AXIS = Enum('AXIS',['loglog', 'xlog', 'ylog','xy'])
RMEM = Enum('RMEM',['p2p', 'put', 'put_trigr', 'put_fast'])
LINESTYLE = Enum('LINESTYLE',['solid', 'dashed'])
MARKSTYLE = Enum('MARKSTYLE',['square', 'star', 'circle'])

# hexa code, no '#' in front!
colors=dict()
colors[RMEM.p2p] = "003f5c"
colors[RMEM.put] = "ffa600"
colors[RMEM.put_trigr] = "bc5090"
colors[RMEM.put_fast] = "bc5090"

idx_list = dict()
idx_list[RMEM.p2p] = 1
idx_list[RMEM.put] = 2
idx_list[RMEM.put_trigr] = 3
idx_list[RMEM.put_fast] = 4

_linewidth = 0.75

def rmem_2_legend(case):
    if(case == RMEM.p2p):
        return "p2p"
    elif(case == RMEM.put):
        return "put"
    elif(case == RMEM.put_trigr):
        return "put trigr"
    elif(case == RMEM.put_fast):
        return "put fast"

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
    
def markstyle_2_tikz(mark):
    if (mark == MARKSTYLE.square):
        return "mark=square"
    elif (mark == MARKSTYLE.star):
        return "mark=*"
    elif (mark == MARKSTYLE.circle):
        return "mark=o"

_color_id = int(0)

def tikz_addplot(tikz_file,x_idx,y_idx, datafile,
            do_injection=False,
            legend="",
            linestyle=LINESTYLE.solid,
            marker=MARKSTYLE.circle):
    global _color_id
    ncolors=len(mpl.cm.get_cmap('tab10').colors)
    #---------------------------------------------------------------------------
    # if(bool(color)):
    color = mpl.cm.get_cmap('tab10').colors[_color_id]
    print(f"color_id = {_color_id} -> color = %{color}")
    tikz_file.write(f"\\definecolor{{color_{_color_id}}}{{RGB}}{{{color[0]*255.0},{color[1]*255.0},{color[2]*255.0}}}\n")
    color_string = f"color_{_color_id}"
    _color_id = (_color_id + 1)%ncolors
    # else:
    #     color_string = ""
    tikz_line = linestyle_2_tikz(linestyle)
    tikz_mark = linestyle_2_tikz(marker)
    #---------------------------------------------------------------------------
    # do it!
    tikz_file.write(f"\\addplot+[{tikz_line},{color_string},line width={_linewidth}pt] table[col sep=comma, x index={x_idx}, y index={y_idx}] {{{datafile}}};\n")
    if(bool(legend)):
        tikz_file.write(f"\\addlegendentryexpanded{{{legend}}};\n")
    #---------------------------------------------------------------------------
    # update the datafile and rerun for injection
    if(do_injection):
        dataf = datafile.replace("r1","r0")
        tikz_line = linestyle_2_tikz(LINESTYLE.dashed)
        tikz_file.write(f"\\addplot+[{tikz_line},{color_string},forget plot,line width={_linewidth}pt] table[col sep=comma, x index={x_idx}, y index={y_idx}] {{{dataf}}};\n")
    

def tikz_header(tikz_file,tikz_opt=""):
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
    tikz_file.write("    {mark=*},\n")
    tikz_file.write("    {mark=square}% <-- don't add a comma here\n")
    tikz_file.write("}\n")
    tikz_file.write("%------------------------------------------------------------------------------------------------\n")
    tikz_file.write("\\pgfplotsset{every axis/.append style={\n")
    tikz_file.write("% grid on\n")
    tikz_file.write("ymajorgrids=true,\n")
    tikz_file.write("xmajorgrids=true,\n")
    tikz_file.write("% grid\n")
    tikz_file.write("grid=both,\n")
    tikz_file.write("log basis x=2,\n")
    tikz_file.write("%minor tick num=2,\n")
    tikz_file.write("%every minor tick/.style={minor tick length=0pt,dashed},\n")
    tikz_file.write("% line width\n")
    tikz_file.write("% position the legend\n")
    tikz_file.write("log ticks with fixed point,\n")
    tikz_file.write("legend pos=north west,\n")
    tikz_file.write("legend cell align={left},\n")
    tikz_file.write("legend style={fill opacity=0.8, draw opacity=1, text opacity=1, draw=lightgray204,font={\\picfont\\tiny}},\n")
    tikz_file.write("mark size = 0.5pt,\n")
    tikz_file.write("cycle list name = colorlist,\n")
    tikz_file.write("cycle multi list = {\n")
    tikz_file.write("	marklist\\nextlist\n")
    tikz_file.write("	colorlist\\nextlist},\n")
    tikz_file.write("},\n")
    # tikz_file.write("every axis plot/.append style={line width=0.5pt}\n")
    if(bool(tikz_opt)):
        tikz_file.write(f"{tikz_opt},\n")
    tikz_file.write("}\n")
    tikz_file.write("\n")
    tikz_file.write("\\begin{document}\n")

def tikz_footer(tikz_file):
    tikz_file.write("\\end{document}\n")



def tikz_axis_open(tikz_file,title,axis=AXIS.xy,tikz_opt=""):
    global _color_id
    _color_id = 0
    tikz_file.write("\\begin{tikzpicture}\n")
    tikz_file.write(f"\\begin{{{axis_2_tikz(axis)}}}[title={{{title}}},\n{tikz_opt}]\n")
def tikz_axis_close(tikz_file,axis=AXIS.xy,break_line=False):
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
            else:
                AssertionError("unknown")
    file.close()
    return key