from src.data_extraction.data_main import Data


class TimeSeriesTool:

    def __init__(self):
        self.data = Data(r'RefData.mat', 'StatClCd.csv', 'StatElev.csv')
        self.mdat = self.data.get_mat().get_data()
        self.pdat = self.data.get_pfd()

        self.clcd_df = self.pdat['StatClCd.csv']
        self.elev_df = self.pdat['StatElev.csv']

        self.clcd_time_list = self.clcd_df["time"].tolist()
        self.elev_time_list = self.elev_df["time"].tolist()

        self.mtime = self.mdat['time']
        self.pheight = self.pdat['StatClCd.csv']['hp']
        self.lhfu = self.mdat['lh_engine_FU']
        self.rhfu = self.mdat['rh_engine_FU']
        self.mtime_flat = self.mtime.flatten()
        self.mtime_list = self.mtime_flat.tolist()
        self.mtime_list_rounded = [round(t, 1) for t in self.mtime_list]

    def get_mdat_tstep_list_idx_for_matching_pdat_tstep(self, pdat_t):
        idx = self.mtime_list_rounded.index(round(pdat_t,1))
        return idx

    def get_t_specific_mdat_values(self, pdat_t):
        pdat_t_idx = self.get_mdat_tstep_list_idx_for_matching_pdat_tstep(pdat_t)
        vars_keys_list = list(self.mdat.keys())
        t_specific_mdat = {}
        for var_key in vars_keys_list:
            t_specific_mdat[var_key] = self.mdat[var_key][pdat_t_idx]
        print("At t= {0} the corresponding recorded 'black-box' data is:\n {1}".format(pdat_t, t_specific_mdat))
        return t_specific_mdat

# if __name__ == "__main__":
#     ts_tool = TimeSeriesTool()
#     t = 5171
#     specific_t_mdat_vals = ts_tool.get_t_specific_mdat_values(t)
#     print("At t= {0} the corresponding recorded 'black-box' data is:\n {1}".format(t, specific_t_mdat_vals))
    # print(ts_tool.get_t_specific_mdat_values(1665))






