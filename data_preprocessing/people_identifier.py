# Load libraries ---------------------------------------------

# ------------------------------------------------------------


class PeopleIdentifier(object):
    
    def __init__(self):
        self.id_column_names = []
        self.pid_cname = ""
        self.next_available_pid = 0
        self.cid_to_pid = {}  # {"col1": {cid1: pid1, cid2: pid2}, "col2":...}
        self.pid_to_cid = {}  # {pid1: {"col1": set(cid1, cid2, ...), "col2": set(...), ...}, pid2: ...}
        self.data = None
        
    def add_pid(self, data, id_column_names, pid_cname):
        self.id_column_names = id_column_names
        self.pid_cname = pid_cname
        
        for cid_cname in id_column_names:
            self.cid_to_pid[cid_cname] = {}
        
        for idx, reservation in data.iterrows():
            pids = set()
            for cid_cname in id_column_names:
                if reservation[cid_cname] in self.cid_to_pid[cid_cname]:
                    pids.add(self.cid_to_pid[cid_cname][reservation[cid_cname]])
#                     print(cid_cname, reservation[cid_cname], self.cid_to_pid[cid_cname][reservation[cid_cname]])
                    
            if len(pids) > 0:
                min_pid = min(pids)
                
                self.set_pid(min_pid, reservation)
                
                # Merge pids connected through this node
                
                if len(pids) > 1:
                    pids.remove(min_pid)
                    self.merge_pids(pids, min_pid)
                    
#                 print("Chosen pid: {}".format(min_pid))
            else:
                new_pid = self.next_available_pid
                self.next_available_pid += 1
                
                self.set_pid(new_pid, reservation)
#                 print("Chosen pid: {}".format(new_pid))
                
#             print("=======")
#             print(self.pid_to_cid)
#             print("=======")
    
        data_pid = data.copy()
        data_pid.loc[:, pid_cname] = data_pid.loc[:, id_column_names[0]].apply(lambda x: self.cid_to_pid[id_column_names[0]][x])
        self.data = data_pid
        
        return data_pid
        
    def set_pid(self, pid, reservation):
        for cid_cname in self.id_column_names:
            if reservation[cid_cname] != "":
                self.cid_to_pid[cid_cname][reservation[cid_cname]] = pid
        if pid in self.pid_to_cid:
            for cid_cname in self.id_column_names:
                self.pid_to_cid[pid][cid_cname] |= {reservation[cid_cname]} if reservation[cid_cname] != "" else set()
        else:
            self.pid_to_cid[pid] = {cid_cname: {reservation[cid_cname]} if reservation[cid_cname] != "" else set() 
                                    for cid_cname in self.id_column_names}
        
    def merge_pids(self, pids_from, pid_to):
        # print("Merge pids", pids_from, pid_to, self.pid_to_cid)
        for pid_from in pids_from:
            for cid_cname in self.id_column_names:
                for cid in self.pid_to_cid[pid_from][cid_cname]:
                    self.cid_to_pid[cid_cname][cid] = pid_to
                self.pid_to_cid[pid_to][cid_cname] |= self.pid_to_cid[pid_from][cid_cname]
            self.pid_to_cid.pop(pid_from)
