import argparse
import glob
import os
import re
import json

import networkx as nx
import numpy as np
from PIL import Image
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
from sigma_graph.data.file_manager import check_dir, find_file_in_dir, load_graph_files
import matplotlib.cm as cm
import matplotlib.colors as mcolors

op_to_dir = {
    'N': (0, 1), 
    'S': (1, 0), 
    'W': (0, -1),
    'E': (-1, 0)
}

def agent_log_parser(line) -> dict:
    agent_info = {}
    # parse team info
    team_red = re.search(r"red:(\d+)", line)
    team_blue = re.search(r"blue:(\d+)", line)
    if team_red is not None:
        agent_info["team"] = "red"
        agent_info["id"] = team_red[1]
    elif team_blue is not None:
        agent_info["team"] = "blue"
        agent_info["id"] = team_blue[1]
    else:
        assert f"[log] Invalid agent team format: {line}"
    # parse agent info
    agent_pos = re.search(r"HP:\s?(\d+) node:(\d+) dir:(\d) pos:\((\d+), (\d+)\)", line)
    if agent_info is not None:
        agent_info["HP"] = int(agent_pos[1])
        agent_info["node"] = int(agent_pos[2])
        agent_info["dir"] = int(agent_pos[3])
        agent_info["pos"] = (int(agent_pos[4]), int(agent_pos[5]))
    else:
        assert f"[log] Invalid agent info format: {line}"
    return agent_info


def list_nums_log_parser(line):
    pass


def log_file_parser(line):
    print(f'line {line}')
    segments = line.split(" | ")
    print(f'segments {segments}')
    step_num = int(re.search(r"Step #\s?(\d+)", segments[0])[1])

    agents = []
    for str_agents in segments[1:-2]:
        agents.append(agent_log_parser(str_agents))
    actions = segments[-2]
    rewards = segments[-1]

    return step_num, agents, actions, rewards[:-1]


def check_log_files(env_dir, log_dir, log_file):
    # generate a subfolder in the log folder for -> animations (and optional pictures for each step)
    log_file_dir = find_file_in_dir(log_dir, log_file)
    fig_file_dir = os.path.join(log_dir, log_file[:-4])
    if not check_dir(fig_file_dir):
        os.mkdir(fig_file_dir)
    return log_file_dir, fig_file_dir


def generate_picture(env_dir, log_dir):
    
    flows_path = f'{log_dir}/flows.json'
    with open(flows_path, "r") as flows_file:
        flows = json.load(flows_file)

    print(f'flows {flows}')

    map_info, _ = load_graph_files(env_path=env_dir, map_lookup="S")
    
    fig = plt.figure()
    fig.patch.set_alpha(0.)
    fig.tight_layout()
    plt.axis('off')
    
    col_map = ["gold"] * len(map_info.n_info)
    
    nx.draw_networkx(map_info.g_acs, map_info.n_info, node_color=col_map, edge_color="blue", arrows=True)

    max_x = max(pos[0] for pos in map_info.n_info.values())
    max_y = max(pos[1] for pos in map_info.n_info.values())
    scale_factor = max(max_x, max_y) / 500

    for node in map_info.g_acs.nodes:
        pos = map_info.n_info[node]
        x, y = pos
        # Node is adjusted for 0 indexed flow log
        dir_flows = flows[str(node-1)]

        color_vals = np.linspace(0, 1, 100)
        colors = [(color, color, color, 0) for color in color_vals]  
        for op in ['N', 'S', 'W', 'E']:
            flow = int((1 - dir_flows[op]) * 99)
            dx, dy = op_to_dir[op]
            plt.arrow(x, y, dx*scale_factor, dy*scale_factor, color=colors[flow], alpha=.8, width=0.02, head_width=0.5)

        circle_radius = 1.25  
        circle_center_x = x + circle_radius  
        circle_center_y = y + circle_radius  
        
        flow = int((1 - dir_flows['NOOP']) * 99)

        circle = plt.Circle((circle_center_x, circle_center_y), circle_radius, color=colors[flow], alpha=0.8, fill=False, linewidth=2)

        plt.gca().add_patch(circle)


    cur_dir = os.getcwd()
    png_file = os.path.join(cur_dir, log_dir, "flows.png")

    plt.savefig(png_file, dpi=100, transparent=True)
    plt.close()



def generate_picture_route(env_dir, log_dir, log_file, route_info):
    log_file_dir, fig_folder = check_log_files(env_dir, log_dir, log_file)
    return fig_folder


def frame_add_background(img_dir, bg_file):

    f = img_dir + "flows.png"

    foreground = Image.open(f)
    background = Image.open(bg_file)
    background.paste(foreground, (0, 0), foreground)

    background.save('flows.jpg', 'JPEG')


def local_run(env_dir, log_dir, prefix, bg_pic, fps, HP_red, TR_red, HP_blue, TR_blue,
              color_decay=True, froze=False, route_only=False, route_info=None):
    directory = os.fsencode(log_dir)
    for file in os.listdir(directory):
        log_file = os.fsdecode(file)
        print(f'log_file {log_file}')
        if log_file.endswith(".txt") and log_file.startswith(prefix):
            if route_only:
                fig_folder = generate_picture_route(env_dir, log_dir, log_file, route_info)
                pause_frame = 0
            else:
                fig_folder, pause_frame = generate_picture(env_dir, log_dir, log_file,
                                                           HP_red, TR_red, HP_blue, TR_blue, color_decay, froze)
            frame_add_background(fig_folder, os.path.join(log_dir, f"{log_file[:-4]}.gif"), bg_pic, fps,
                                 pause_frame)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env_dir', type=str, default='../../', help='path to project root')
    # parser.add_argument('--log_dir', type=str, default='../../logs/visuals/demo/', help='path to log file folder')
    # parser.add_argument('--prefix', type=str, default='log_', help='log file prefix')
    # parser.add_argument('--background', type=str, default='../../logs/visuals/background.png')
    # parser.add_argument('--fps', type=int, default=2)  # frame per second in animations

    # parser.add_argument('--HP_froze_on', action='store_true', default=False, help='stop animation if agent is dead')
    # parser.add_argument('--HP_red', type=int, default=100)
    # parser.add_argument('--TR_red', type=int, default=5)
    # parser.add_argument('--HP_blue', type=int, default=100)
    # parser.add_argument('--TR_blue', type=int, default=10)
    # parser.add_argument('--HP_color_off', action='store_false', default=True, help='gradient colors for HP')

    # parser.add_argument('--route_only', type=bool, default=False)  # exclude step info
    # parser.add_argument('--route_info', type=str, default='name')  # choose from ['name', 'pos', 'idx']
    # args = parser.parse_args()

    # local_run(args.env_dir, args.log_dir, args.prefix, args.background, args.fps,
    #           args.HP_red, args.TR_red, args.HP_blue, args.TR_blue, args.HP_color_off,
    #           args.HP_froze_on, args.route_only, args.route_info)
    env_dir = '../'
    log_dir = 'logs/temp'
    log_file = 'flows.json'
    HP_red = 100
    TR_red = 5
    HP_blue = 100
    TR_blue = 10
    generate_picture(env_dir, log_dir)
    fig_folder = "."

    # frame_add_background(fig_folder, os.path.join(log_dir, f"{log_file[:-4]}.gif"), bg_pic, fps, pause_frame)