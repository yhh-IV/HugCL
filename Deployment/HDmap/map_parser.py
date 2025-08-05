import numpy as np 
import numpy.linalg as npl
import xml.etree.ElementTree as xml
import matplotlib.pyplot as plt 

def polyline_length(line):
    '''
    line: np.array
    '''
    
    if len(line) <= 1:
        return 0
    
    dist_list = np.cumsum(np.linalg.norm(np.diff(line, axis = 0), axis = 1))

    return dist_list[-1]

def dense_polyline2d(line, resolution=0.1):
    """
    Dense a polyline by linear interpolation.

    :param resolution: the gap between each point should be lower than this resolution
    :param interp: the interpolation method
    :return: the densed polyline
    """

    if line is None or len(line) == 0:
        raise ValueError("Line input is null")

    s = np.cumsum(npl.norm(np.diff(line, axis=0), axis=1))
    s = np.concatenate([[0], s])
    num = int(round(s[-1]/resolution))

    try:
        s_space = np.linspace(0, s[-1], num = num)
    except:
        raise ValueError(num, s[-1], len(s))

    x = np.interp(s_space,s,line[:,0])
    y = np.interp(s_space,s,line[:,1])

    return np.array([x,y]).T
    
def parse_osm(filename):
    e = xml.parse(filename).getroot()
    global point_dict
    point_dict = dict()
    for node in e.findall("node"):
        for tag in node.findall("tag"):
            if tag.get('k') == 'local_x':
                x = float(tag.get('v'))
            if tag.get('k') == 'local_y':
                y = float(tag.get('v'))
        point_dict[int(node.get('id'))] = [x, y]

    way_dict = dict()
    left_way_ids = [7899, 9095, 4286, 9119, 9123, 9138, 7898]
    right_way_ids = [7901, 9096, 4287, 3992, 9130, 9139, 7900]
    for way in e.findall("way"):
        way_id = int(way.get('id'))
        if way_id not in (left_way_ids + right_way_ids):
            continue
        
        sub_dict = dict()
        ref_list = []
        for way_node in way.findall("nd"):
            ref_list.append(int(way_node.get('ref')))

        point_list = [point_dict[ref] for ref in ref_list]
        sub_dict['traj'] = np.array(point_list)
        
        for way_node in way.findall("tag"):
            if way_node.get('k') == "type":
                sub_dict['type'] = way_node.get('v')
            if way_node.get('k') == "subtype":
                sub_dict['subtype'] = way_node.get('v')
        way_dict[int(way.get('id'))] = sub_dict
    
    for sub_id, sub_dict in way_dict.items():
        way_traj = np.column_stack((sub_dict['traj'][:, 0], sub_dict['traj'][:, 1]))
        dense_way_traj = dense_polyline2d(way_traj)
        sub_dict["dense_traj"] = dense_way_traj
        plt.plot(dense_way_traj[:, 0], dense_way_traj[:, 1], linewidth=1, color='black', linestyle=sub_dict['subtype'], zorder=2)
        plt.gca().set_aspect('equal')
        
    # global ctl_dict # center line
    global ctl_dict
    ctl_dict = dict()
    for idx in range(len(left_way_ids)):
        ref_dict = dict()
        ref_dict["left_id"] = left_way_ids[idx]
        ref_dict["right_id"] = right_way_ids[idx]
        ctl_dict[idx] = ref_dict 
    
    # calculate center line
    for relation in ctl_dict.keys():
        left_ref_id = ctl_dict[relation]["left_id"]
        left_ref = way_dict[int(left_ref_id)]["dense_traj"]
        right_ref_id = ctl_dict[relation]["right_id"]
        right_ref = way_dict[int(right_ref_id)]["dense_traj"]
        if len(left_ref) < len(right_ref):
            base_ref = left_ref
            search_ref = right_ref
        else:
            base_ref = right_ref
            search_ref = left_ref
            
        center_line = []
        for base_point in base_ref:
            dists = np.linalg.norm(search_ref - base_point, axis=1)
            idx = np.argmin(dists)
            mid_point = (base_point + search_ref[idx]) / 2
            center_line.append(mid_point)
        ctl_dict[relation]["cl"] = dense_polyline2d(np.array(center_line))
    
    ref_path(ctl_dict)
    plt.gca().set_aspect('equal')
   

def ref_path(ctl_dict):
    ref_path = {
    0: {'lane_0': ctl_dict[0]["cl"], 'lane_1': ctl_dict[1]["cl"]}, # 0: left 1: right
    1: {'lane_0': np.vstack((ctl_dict[2]["cl"], np.flip(ctl_dict[3]["cl"], axis=0), np.flip(ctl_dict[4]["cl"], axis=0)))},
    2: {'lane_0': np.vstack((ctl_dict[5]["cl"], ctl_dict[6]["cl"]))},
}

    ### check ref_path ###
    for seg_id in range(len(ref_path)):
        for lane_id in ref_path[seg_id].keys():
            plt.plot(ref_path[seg_id][lane_id][:, 0], ref_path[seg_id][lane_id][:, 1], linewidth=1, color='gray', linestyle='--')
    
if __name__ == '__main__':
    path = './icv.osm'
    parse_osm(path)
