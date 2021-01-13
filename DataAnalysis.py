import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
import matplotlib.pyplot as pl
import tkinter as tk
from tkinter import filedialog, dialog
import os


class Data(object):

    def __init__(self, disp_list, force_list, data_type=None, reincr=0.01, reprotocol=None):
        """
        :param disp_list: displacement list
        :param force_list: force list
        :param data_type: choose from "mono", "cyc_half" and "cyc_full"
        :param reincr: increment for re-organizing data (default = 0.1)
        :param reprotocol: pre-defined protocol for re-organizing data (obtained from curve as default)
        """
        self.disp_list = disp_list
        self.force_list = force_list
        self.true_data_type = data_type
        self.incr = reincr
        self.protocol = reprotocol
        self.maxdisp = max(self.disp_list)
        self.mindisp = min(self.disp_list)

    @staticmethod
    def reorg(data, set_range=None, incr=None, datanumber=100.0):
        """
        :param data: original data
        :param set_range: pre-defined x-range of re-organized data (default = None)
        :param incr: pre-defined x-increment of re-organized data (default = None)
        :param datanumber: pre-defined data number of re-organized data if incr is not set (default = 100)
        :return: re-organized data, in form of [reorg_data_x, reorg_data_y]
        """
        # set new range for x data
        reorg_range = set_range.copy() if set_range else [data[0][0], data[0][-1]]
        # set increment for x data
        reorg_incr = incr if incr else abs(reorg_range[0] - reorg_range[1]) / datanumber
        # check the sign of the increment
        reorg_incr = -reorg_incr if reorg_range[0] >= reorg_range[1] else reorg_incr
        # re-organize data
        f = interp1d(data[0], data[1], fill_value='extrapolate')
        reorg_data_x = list(np.arange(reorg_range[0], reorg_range[1], reorg_incr))
        reorg_data_y = list(f(reorg_data_x))
        reorg_data = [reorg_data_x, reorg_data_y]
        return reorg_data

    @staticmethod
    def cross_point(line1, line2):
        """
        :param line1: the first line
        :param line2: the second line
        :return: the cross point lists of the two lines
        """
        cross_point_list = []
        lower_bound, upper_bound = min(line1[0]+line2[0]), max(line1[0]+line2[0])*1.5
        incr = (upper_bound - lower_bound) / 10000.0
        f1 = interp1d(line1[0], line1[1], fill_value='extrapolate')
        f2 = interp1d(line2[0], line2[1], fill_value='extrapolate')
        line1_x_list = list(np.arange(lower_bound, upper_bound, incr))
        line1_y_list = list(f1(line1_x_list))
        line2_x_list = list(np.arange(lower_bound, upper_bound, incr))
        line2_y_list = list(f2(line2_x_list))
        for index in range(len(line1_x_list) - 1):
            if (line2_y_list[index + 1] - line1_y_list[index + 1]) * (line2_y_list[index] - line1_y_list[index]) <= 0:
                cross_point = [line1_x_list[index], line1_y_list[index]]
                cross_point_list.append(cross_point)
        return cross_point_list

    @staticmethod
    def tangent_point(line, slope, direction='top'):
        """
        :param line: target line
        :param slope: slope of the tangent line
        :param direction: direction from tangent line to target line (default = 'top')
        :return: tangent point lists on the target line
        """
        tangent_point = []
        for index in range(len(line[0]) - 2):
            k1 = (line[1][index + 1] - line[1][index]) / (line[0][index + 1] - line[0][index])
            k2 = (line[1][index + 2] - line[1][index + 1]) / (line[0][index + 2] - line[0][index + 1])
            if direction.lower() in 'both':
                if (k2 - slope) * (k1 - slope) <= 0:
                    tangent_point.append([line[0][index], line[1][index]])
            elif direction.lower() in 'bottom':
                if k1 <= slope <= k2:
                    tangent_point.append([line[0][index], line[1][index]])
            elif direction.lower() in 'top':
                if k1 >= slope >= k2:
                    tangent_point.append([line[0][index], line[1][index]])
            else:
                raise Exception('direction not set correctly')
        if tangent_point is []:
            raise Exception('Can not find such point')
        return tangent_point

    @staticmethod
    def plotitem(basics, addition=None):
        """
        :param basics: basic information to be plotted
        :param addition: additional information to be plotted
        :return: Plot without returning results
        """
        # determine item list
        item_list = basics + addition if addition else basics
        # plot basics information
        for item in item_list:
            pl.plot(item[0], item[2], label=item[4], color=item[5], marker=item[6])
            pl.xlabel(item[1])
            pl.ylabel(item[3])
        pl.legend()
        pl.show()

    @staticmethod
    def saveitem(item_name_list, item_list, filename):
        """
        :param item_list:
        :param filename:
        :return:
        """
        with open('%s.txt'%filename, "w") as f:
            for item_name in item_name_list:
                f.write("%s " % item_name)
            f.write("\n")
            for line_number in range(len(item_list[0])):
                for item in item_list:
                    f.write('%.f '%item[line_number])
                f.write('\n')
        f.close()

    @property
    def data_type(self):
        """
        :return: data type
        """
        if self.true_data_type:
            self.true_data_type = self.true_data_type
        else:
            if self.maxdisp/(self.maxdisp-self.mindisp) >= 0.1 and self.mindisp/(self.maxdisp-self.mindisp) <= -0.1:
                self.true_data_type = 'cyc_full'
            else:
                index_keypoint = 0
                for index in range(len(self.disp_list)-1):
                    if self.disp_list[index]*self.disp_list[index+1] <= 0:
                        index_keypoint = index
                self.true_data_type = 'mono' if index_keypoint <= 0.1 * len(self.disp_list) else 'cyc_half'
        return self.true_data_type

    def plotcurve(self):
        """
        :return: plot [disp_list, force_list] without returning results
        """
        basics = [[self.disp_list, 'deformation', self.force_list, 'force', 'original curve', "k", None]]
        self.plotitem(basics)

    def plotprocedure(self, plottime=60):
        """
        :param plottime: time (seconds) for data plot
        :return:
        """
        # modify data to match plot time
        datanumber = len(self.disp_list)
        filternum = int(datanumber / plottime * 0.05)
        modisp_list, moforce_list = [], []
        for i in range(datanumber):
            if filternum == 0:
                filternum = 1
            if i % filternum == 0:
                modisp_list.append(self.disp_list[i])
                moforce_list.append(self.force_list[i])
        modisp_s = [modisp_list[0], modisp_list[1]]
        moforce_s = [moforce_list[0], moforce_list[1]]
        max_disp, min_disp= modisp_list[0], modisp_list[0]
        max_force, min_force = moforce_list[0], moforce_list[0]
        # plot
        pl.xlabel('disp')
        pl.ylabel('force')
        for i in range(len(moforce_list) - 1):
            # update the two points
            modisp_s[0], modisp_s[1] = modisp_list[i], modisp_list[i + 1]
            moforce_s[0], moforce_s[1] = moforce_list[i], moforce_list[i + 1]
            min_disp, max_disp = min(modisp_s[1], min_disp), max(modisp_s[1], max_disp)
            min_force, max_force = min(moforce_s[1], min_force), max(moforce_s[1], max_force)
            pl.plot(modisp_s, moforce_s, "b")
            pl.xlim(min_disp * 1.2, max_disp * 1.2)
            pl.ylim(min_force * 1.2, max_force * 1.2)
            pl.pause(0.05)
        pl.show()

    def cyc_index_list(self):
        """
        :return: index list for the first point of each cycle, including index of the last point in the last cycle (i+1)
        """
        cyc_index_list = [0]
        if self.data_type.lower() in "mono":
            cyc_index_list.append(len(self.disp_list)-1)
        elif self.data_type.lower() in "cyc_half":
            for i in range(len(self.disp_list) - 1):
                if self.disp_list[i] > 0 >= self.disp_list[i + 1] and i - cyc_index_list[-1] > 3:
                    cyc_index_list.append(i+1)
        elif self.data_type.lower() in "cyc_full":
            for i in range(len(self.disp_list) - 1):
                if self.disp_list[i] < 0 <= self.disp_list[i + 1] and i - cyc_index_list[-1] > 3:
                    cyc_index_list.append(i+1)
        else:
            raise Exception('Data type is not correct')
        # last point included
        cyc_index_list[-1] += 1
        return cyc_index_list

    def cycle_no(self):
        """
        :return: count of all cycles
        """
        cycle_no = len(self.cyc_index_list()) - 1
        return cycle_no

    def data_cycle(self, cycle):
        """
        :param cycle: cycle number
        :return: [disp_list,force_list] in ith cycle
        """
        cyc_index_list = self.cyc_index_list()
        if cycle <= self.cycle_no():
            begin = cyc_index_list[cycle - 1]
            end = cyc_index_list[cycle]
            disp_list_cycle = self.disp_list[begin:end]
            force_list_cycle = self.force_list[begin:end]
        else:
            raise Exception('Cycle number out of range')
        return [disp_list_cycle, force_list_cycle]

    def obtain_protocol(self, plotopt=None, saveopt=None):
        """
        :param plotopt: determine if plot a figure
        :param saveopt: determine if save a text file
        :return: protocol for the input data, in form of [cycle_list, protocol_list, proindex_list]
        """
        protocol, procycle, proindex = [0, ], [0, ], [0, ]
        cyc_index_list = self.cyc_index_list()
        if self.data_type.lower() in "mono":
            protocol.append(max(self.disp_list))
            procycle.append(1)
            proindex.append(len(self.disp_list)-1)
        elif self.data_type.lower() in "cyc_half":
            for cycle_number in range(1, self.cycle_no()+1):
                disp_list_cycle = self.data_cycle(cycle_number)[0]
                maxdisp_cycle = max(disp_list_cycle)
                maxdisp_cycle_index = disp_list_cycle.index(maxdisp_cycle) + cyc_index_list[cycle_number-1]
                protocol += [maxdisp_cycle, 0.0]
                procycle += [cycle_number - 0.5, cycle_number]
                proindex += [maxdisp_cycle_index, cyc_index_list[cycle_number]]
        elif self.data_type.lower() in "cyc_full":
            for cycle_number in range(1, self.cycle_no()+1):
                disp_list_cycle = self.data_cycle(cycle_number)[0]
                maxdisp_cycle, mindisp_cycle = max(disp_list_cycle), min(disp_list_cycle)
                maxdisp_cycle_index = disp_list_cycle.index(maxdisp_cycle) + cyc_index_list[cycle_number-1]
                mindisp_cycle_index = disp_list_cycle.index(mindisp_cycle) + cyc_index_list[cycle_number-1]
                protocol += [max(disp_list_cycle), min(disp_list_cycle), 0.0]
                procycle += [cycle_number - 0.75, cycle_number - 0.25, cycle_number]
                proindex += [maxdisp_cycle_index, mindisp_cycle_index, cyc_index_list[cycle_number]]
        else:
            raise Exception('Data type is not correct')
        proindex[-1] += 1
        # determine whether plot a figure
        if plotopt:
            basics = [[procycle, 'cycle number', protocol, 'deformation', 'Protocol', 'k', None]]
            self.plotitem(basics)
        # determine if save a text file
        if saveopt:
            item_name_list, item_list, filename = ['cycle number', 'disp'], [procycle, protocol], 'protocol'
            self.saveitem(item_name_list, item_list, filename)
        return [procycle, protocol, proindex]

    def reorgdata(self, plotopt=None, saveopt=None):
        """
        :param plotopt: determine if plot a figure
        :param saveopt: determine if save a text file
        :return: re-organized data, in form of [disp_list_new, force_list_new]
        """
        # set protocol
        protocol = self.protocol if self.protocol else self.obtain_protocol()[1]
        proindex = self.obtain_protocol()[2]
        # declare variables
        reorg_disp_list, reorg_force_list = [], []
        # re-organized data
        for i in range(len(protocol)-1):
            disp_sec = self.disp_list[proindex[i]:proindex[i+1]]
            force_sec = self.force_list[proindex[i]:proindex[i+1]]
            range_sec = [protocol[i], protocol[i+1]]
            reorg_data_sec = self.reorg([disp_sec, force_sec], set_range=range_sec, incr=self.incr)
            reorg_disp_list += reorg_data_sec[0]
            reorg_force_list += reorg_data_sec[1]
        # determine whether plot a figure
        if plotopt:
            basics = [[self.disp_list, 'deformation', self.force_list, 'force', 'original data', 'k', None],
                      [reorg_disp_list, 'deformation', reorg_force_list, 'force', 're-organized data', "r", None]]
            self.plotitem(basics)
        # determine if save a text file
        if saveopt:
            item_name_list, item_list, filename = ['disp', 'force'], [reorg_disp_list, reorg_force_list], \
                                                  're-organized data'
            self.saveitem(item_name_list, item_list, filename)
        return [reorg_disp_list, reorg_force_list]

    def backbone(self, plotopt=None, saveopt=None):
        """
        :param plotopt: determine if plot a figure
        :param saveopt: determine if save a text file
        :return: backbone curve, in form of [disp_list,force_list]
        """
        if self.data_type.lower() in "mono":
            raise Exception('Not cyclic data')
        elif self.data_type.lower() in "cyc_half" or "cyc_full":
            backbonedisp_list, backboneforce_list = [0], [0]
            maxdisp_hist, mindisp_hist = 0, 0
            for cycle in range(1, self.cycle_no() + 1):
                cycledisp_list, cycleforce_list = self.data_cycle(cycle)[0], self.data_cycle(cycle)[1]
                if max(cycledisp_list) > maxdisp_hist:
                    maxdisp_hist = 1.05 * max(cycledisp_list)
                    for cycledisp, cycleforce in zip(cycledisp_list, cycleforce_list):
                        # add point corresponding to max force in this cycle
                        if cycleforce == max(cycleforce_list):
                            backboneforce_list.append(cycleforce)
                            backbonedisp_list.append(cycledisp)
                        if 0.95 * max(self.disp_list) <= cycledisp <= 1.05 * max(self.disp_list) and cycledisp == max(cycledisp_list):
                            backboneforce_list.append(cycleforce)
                            backbonedisp_list.append(cycledisp)
                if min(cycledisp_list) < mindisp_hist:
                    mindisp_hist = 1.05 * min(cycledisp_list)
                    for cycledisp, cycleforce in zip(cycledisp_list, cycleforce_list):
                        # add point corresponding to min force in this cycle
                        if cycleforce == min(cycleforce_list):
                            backboneforce_list.insert(0, cycleforce)
                            backbonedisp_list.insert(0, cycledisp)
                        if 1.05 * min(self.disp_list) <= cycledisp <= 0.95 * min(self.disp_list) and cycledisp == min(cycledisp_list):
                            backboneforce_list.insert(0, cycleforce)
                            backbonedisp_list.insert(0, cycledisp)
            backbone = [backbonedisp_list, backboneforce_list]
        else:
            raise Exception('Data type is not correct')
        # determine whether plot a figure
        if plotopt:
            basics = [[self.disp_list, 'deformation', self.force_list, 'force', 'Hysteresis', 'k', None],
                      [backbone[0], 'deformation', backbone[1], 'force', 'backbone', 'r', '*']]
            self.plotitem(basics)
        # determine if save a text file
        if saveopt:
            item_name_list, item_list, filename = ['disp', 'force'], backbone, 'backbone'
            self.saveitem(item_name_list, item_list, filename)
        return backbone

    def energy(self, cumulative=None, plotopt=None, saveopt=None):
        """
        :param cumulative: decide whether the cumulative energy is considered
        :param plotopt: determine if plot a figure
        :param saveopt: determine if save a text file
        :return: Energy dissipated in each cycle (or Energy cumulated for each cycle),
        in form of [cycle_list,energy_list]
        """
        # declare vars
        cycle_list = list(range(self.cycle_no() + 1))
        energy_list = [0]
        for cycle in range(1, self.cycle_no() + 1):
            cycledisp_list, cycleforce_list = self.data_cycle(cycle)[0], self.data_cycle(cycle)[1]
            energy = integrate.trapz(cycleforce_list, cycledisp_list)
            if cumulative:
                energy_list.append(energy + energy_list[-1])
            else:
                energy_list.append(energy)
        # determine whether plot a figure
        if plotopt:
            label = 'cumulative energy' if cumulative else 'energy in cycle'
            basics = [[cycle_list, 'cycle number', energy_list, 'energy', label, "k", None]]
            self.plotitem(basics)
        # determine if save a text file
        if saveopt:
            item_name_list = ['cycle number', 'cumulative energy'] if cumulative else ['cycle no', 'energy in cycle']
            item_list = [cycle_list, energy_list]
            filename = 'cumulative energy' if cumulative else 'energy in cycle'
            self.saveitem(item_name_list, item_list, filename)
        return [cycle_list, energy_list]

    def yield_point(self, method, cyc_trans='positive', disp_range=None, plotopt=None, saveopt=None):
        """
        :param method: method for calculating yield point (opt: 'yk', 'eeep', 'cen', 'kc', 'csiro')
        :param cyc_trans: determine if transfer side of backbone curve for cyc_full data
                          (default = 'positive', opt:'negative')
        :param disp_range: pre-defined range for the yield point
        :param plotopt: determine if plot a figure
        :param saveopt: determine if save a text file
        :return: yield_point, in form of [disp, force]
        """
        # determine curve
        curve = None
        if self.data_type.lower() in "mono":
            curve = [self.disp_list, self.force_list]
        elif self.data_type.lower() in "cyc_half":
            curve = self.backbone().copy()
        elif self.data_type.lower() in "cyc_full":
            curve_total = self.backbone().copy()
            if cyc_trans.lower() in "positive":
                index = curve_total[0].index(0)
                curve = [self.backbone()[0][index:], self.backbone()[1][index:]]

            elif cyc_trans.lower() in "negative":
                index = curve_total[0].index(0)
                curve_negative = [self.backbone()[0][:index + 1], self.backbone()[1][:index + 1]]
                curve_negative[0].reverse()
                curve_negative[1].reverse()
                curve = [[-disp for disp in curve_negative[0]], [-force for force in curve_negative[1]]]
            else:
                raise Exception('cyc_trans not correct')
        # re-organize curve
        curve = self.reorg(curve, incr=self.incr)
        # > calculate yield point
        # declare vars
        yield_point = [None, None]
        basics = []
        addition = []
        # y&k method
        if method.lower() in "y&k" or "yk":
            maxforce, maxdisp = max(self.force_list), max(self.disp_list)
            force_0d1, force_0d4 = maxforce * 0.1, maxforce * 0.4
            line_0d1 = [[0, maxdisp], [force_0d1, force_0d1]]
            line_0d4 = [[0, maxdisp], [force_0d4, force_0d4]]
            point_0d1 = self.cross_point(line_0d1, curve)[0]
            point_0d4 = self.cross_point(line_0d4, curve)[0]
            k1 = (point_0d4[1] - point_0d1[1]) / (point_0d4[0] - point_0d1[0])
            line_alpha = [[point_0d1[0], point_0d4[0]], [point_0d1[1], point_0d4[1]]]
            k2 = 1.0 / 6.0 * k1
            tangent_point_list = self.tangent_point(curve, k2)
            line_beta = None
            if disp_range:
                for tangent_point in tangent_point_list:
                    line_beta = [[0, tangent_point[0]], [tangent_point[1] - tangent_point[0] * k2, tangent_point[1]]]
                    possible_point = self.cross_point(line_alpha, line_beta)[0]
                    if disp_range[0] <= possible_point[0] <= disp_range[1]:
                        yield_point = possible_point.copy()
                        break
            else:
                tangent_point = tangent_point_list[0]
                line_beta = [[0, tangent_point[0]], [tangent_point[1] - tangent_point[0] * k2, tangent_point[1]]]
                yield_point = self.cross_point(line_alpha, line_beta)[0]
            # additional plot option
            addition = [
                [line_alpha[0] + [yield_point[0]], '', line_alpha[1] + [yield_point[1]], '', '', 'b', None],
                [line_beta[0], '', line_beta[1], '', '', 'b', None]]
        elif method.lower() in "eeep":
            pass
        else:
            raise Exception('Yield calculating method not exists')
        # determine whether plot a figure
        if plotopt:
            if self.data_type.lower() in "mono":
                basics.append([self.disp_list, 'deformation', self.force_list, 'force', 'Mono Curve', 'k', None],)
            else:
                basics.append([self.disp_list, 'deformation', self.force_list, 'force', 'Hysteresis', 'k', None],)
                basics.append([self.backbone()[0], 'deformation', self.backbone()[1], 'force', 'backbone', "r", "*"])
            basics.append([[yield_point[0]], '', [yield_point[1]], '', 'yield point', "g", "o"])
            self.plotitem(basics, addition=addition)
        # determine if save a text file
        if saveopt:
            item_name_list, item_list, filename= ['yield disp', 'yield force'], yield_point, 'yield point'
            self.saveitem(item_name_list, item_list, filename)
        return yield_point


# if __name__ == '__main__':
    # window = tk.Tk()
    # window.title('窗口标题')  # 标题
    # window.geometry('500x500')  # 窗口尺寸
    #
    # file_path = ''
    #
    # file_text = ''
    #
    # text1 = tk.Text(window, width=50, height=10, bg='orange', font=('Arial', 12))
    # text1.pack()
    #
    # def open_file():
    #     """
    #     :return:
    #     """
    #     global file_path
    #     global file_text
    #     file_path = filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser('H:/')))
    #     print('打开文件：', file_path)
    #     if file_path is not None:
    #         with open(file=file_path, mode='r+', encoding='utf-8') as file:
    #             file_text = file.read()
    #         text1.insert('insert', file_text)
    #
    #
    # def save_file():
    #     global file_path
    #     global file_text
    #     file_path = filedialog.asksaveasfilename(title=u'保存文件')
    #     print('保存文件：', file_path)
    #     file_text = text1.get('1.0', tk.END)
    #     if file_path is not None:
    #         with open(file=file_path, mode='a+', encoding='utf-8') as file:
    #             file.write(file_text)
    #         text1.delete('1.0', tk.END)
    #         dialog.Dialog(None, {'title': 'File Modified', 'text': '保存完成', 'bitmap': 'warning', 'default': 0,
    #                              'strings': ('OK', 'Cancle')})
    #         print('保存完成')
    #
    #
    # bt1 = tk.Button(window, text='打开文件', width=15, height=2, command=open_file)
    # bt1.pack()
    # bt2 = tk.Button(window, text='保存文件', width=15, height=2, command=save_file)
    # bt2.pack()
    #
    # window.mainloop()  # 显示
