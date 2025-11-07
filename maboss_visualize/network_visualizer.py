import json
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm
import os
import shutil

import matplotlib.pyplot as plt
import plotly.graph_objects as go


### Load dataset ###
def load_cyjs(network_path):
    # Load the .cyjs file
    with open(network_path, 'r') as f:
        cyjs_data = json.load(f)

    # Create an empty NetworkX graph
    G = nx.Graph()

    # Add nodes to the graph using 'name' instead of 'id'
    for node in cyjs_data['elements']['nodes']:
        data = node['data']
        node_name = data.get('name')
        
        if node_name:
            if 'position' in node:
                pos = node['position']
                G.add_node(node_name, **data, x=pos.get('x'), y=pos.get('y'))
            else:
                G.add_node(node_name, **data)

    # Add edges to the graph using 'name' instead of 'id'
    for edge in cyjs_data['elements']['edges']:
        data = edge['data']
        source = data['source']
        target = data['target']
        
        source_name = next((n['data']['name'] for n in cyjs_data['elements']['nodes'] if n['data']['id'] == source), None)
        target_name = next((n['data']['name'] for n in cyjs_data['elements']['nodes'] if n['data']['id'] == target), None)
        
        if source_name and target_name:
            G.add_edge(source_name, target_name, **data)

    # Extract node positions for plotting using 'name'
    pos = {node: (data['x'], -data['y']) for node, data in G.nodes(data=True) if 'x' in data and 'y' in data}

    # Define edge colors and arrow styles (Example: based on an attribute or a custom list)
    edge_colors = []
    edge_styles = []
    for edge in G.edges(data=True):
        # Example: Assuming 'color' and 'style' are attributes in the edge data
        #color = edge[2].get('color', '#888')  # Default color if not provided
        if edge[2]['interaction'] == 'inhibit':
            color = '#ff5733'
        if edge[2]['interaction'] == 'activate':
            color = '#156a03'
        style = edge[2].get('style', 'solid')  # Default style if not provided
        edge_colors.append(color)
        edge_styles.append(style)
    print('Network loaded : ' + network_path)

    return(G,pos,edge_colors,edge_styles)

def load_xgmml(network_path):
    # Initialize an empty directed graph
    G = nx.DiGraph()

    # Parse the XGMML file
    tree = ET.parse(network_path)
    root = tree.getroot()

    # Namespace dictionary (if needed)
    ns = {'xgmml': 'http://www.cs.rpi.edu/XGMML'}

    # Iterate through all nodes in the XGMML
    for node in root.findall('.//xgmml:node', ns):
        node_label = node.get('label')
        
        # Use the label as the node ID
        G.add_node(node_label, label=node_label)
        
        # Add any custom attributes
        for att in node.findall('xgmml:att', ns):
            att_name = att.get('name')
            att_value = att.get('value')
            G.nodes[node_label][att_name] = att_value
            
        # Parse the node position (x, y, z) from the graphics element
        graphics = node.find('xgmml:graphics', ns)
        if graphics is not None:
            x = float(graphics.get('x', 0.0))
            y = float(graphics.get('y', 0.0))
            z = float(graphics.get('z', 0.0))  # Optional, default to 0.0 if not present
            G.nodes[node_label]['pos'] = (x, y, z)

    # Iterate through all edges in the XGMML
    for edge in root.findall('.//xgmml:edge', ns):
        source_id = edge.get('source')
        target_id = edge.get('target')
        
        # Find the corresponding node labels for the source and target
        source_label = root.find(f".//xgmml:node[@id='{source_id}']", ns).get('label')
        target_label = root.find(f".//xgmml:node[@id='{target_id}']", ns).get('label')
        
        edge_label = edge.get('label')
        
        # Add the edge with the labels as identifiers
        G.add_edge(source_label, target_label, label=edge_label)
        
        # Add any custom attributes
        for att in edge.findall('xgmml:att', ns):
            att_name = att.get('name')
            att_value = att.get('value')
            G.edges[source_label, target_label][att_name] = att_value

    pos = {node: (data['pos'][0], -data['pos'][1]) for node, data in G.nodes(data=True)}

    # Define edge colors and arrow styles (Example: based on an attribute or a custom list)
    edge_colors = []
    edge_styles = []
    for edge in G.edges(data=True):
        # Example: Assuming 'color' and 'style' are attributes in the edge data
        #color = edge[2].get('color', '#888')  # Default color if not provided
        if edge[2]['interaction'] == 'inhibit':
            color = '#ff5733'
        if edge[2]['interaction'] == 'activate':
            color = '#156a03'
        style = edge[2].get('style', 'solid')  # Default style if not provided
        edge_colors.append(color)
        edge_styles.append(style)
    print('Network loaded : ' + network_path)

    return(G,pos,edge_colors,edge_styles)

class Net_visualizer:

    ### Load Network ###
    def __init__(self):
        self.network_path = np.nan
    
    def load_network_cyjs(self, network_path):
        self.network_path = network_path
        self.network,self.position,self.edge_colors,self.edge_styles = load_cyjs(self.network_path)
    
    def load_network_xgmml(self, network_path):
        self.network_path = network_path
        self.network,self.position,self.edge_colors,self.edge_styles = load_xgmml(self.network_path)

    def load_activity(self, activity):
        self.activity = activity

    def load_timeseries(self, timeseries):
        self.timeserie = timeseries

    ### Plot functions ###
    #### Static Plots ####
    def plot_network(self, fig_size = (12,10)):
        G = self.network
        pos = self.position
        edge_colors = self.edge_colors

        # Plot the network using the positions and 'name' as node identifiers
        plt.figure(figsize=fig_size)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        nx.draw_networkx_edges(G,pos,arrows = True,arrowstyle="<-",arrowsize=10,edge_color=edge_colors,width=1)
        plt.show() 
    
    def plot_network_activity(self, timepoint = 0, node_size = 800, figure_size = (20, 15)):
        """
        Plots the network activity at a given timepoint.

        Parameters:
        -----------
        timepoint : int, optional
            The timepoint at which to visualize the network activity. Default is 0.
        node_size : int, optional
            The base size of the nodes in the network. Default is 800.
        figure_size : tuple, optional
            The size of the figure to be created. Default is (20, 15).

        Returns:
        --------
        None
        """
        # Load dataset
        G = self.network
        pos = self.position
        edge_colors = self.edge_colors
        df_activity = self.timeserie
        nodesize = node_size

        # Ensure the node_size array has the same length as the number of nodes in the graph
        node_sizes = nodesize * np.array([df_activity.loc[timepoint].get(node, 0) for node in G.nodes])

        # Normalize the node values for color mapping
        norm = plt.Normalize(vmin=0, vmax=1)
        node_colors = cm.viridis(norm([df_activity.loc[timepoint].get(node, 0) for node in G.nodes]))

        # Plot the network using the positions and 'name' as node identifiers
        plt.figure(figsize=figure_size)
        options = {
            "node_size": node_sizes,
            "node_color": node_colors,
            "edgecolors": "black",
            "linewidths": 2,
            'arrowstyle':'-'
        }
        nx.draw_networkx(G, pos, **options, with_labels=False)
        nx.draw_networkx_labels(G, pos, 
                                font_size=12, font_family='sans-serif',
                                clip_on=False,
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha = .5),
                                #bbox_to_anchor=(0.5, 0.5),
                                verticalalignment='bottom')
        nx.draw_networkx_edges(G,pos,
                            arrows = True,
                            arrowstyle="->",
                            arrowsize=15, 
                            edge_color=edge_colors, width=2)

        # Create legend for node sizes
        sizes = [nodesize*1/4, nodesize*1/2, nodesize*3/4, nodesize]
        for size in sizes:
            plt.scatter([], [], s=size, c='gray', alpha=0.5, label=str(size/800))
        
        # Create a colorbar for the node activity
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), shrink=0.5, aspect=10, anchor=(-0.1, 0.8), location='right')
        cbar.set_label('Node Activity', rotation=270, labelpad=18, fontsize = 15)

        plt.legend(scatterpoints=1, 
                frameon=True, labelspacing=1, 
                title='Node Activity', loc='lower right', edgecolor='black', bbox_to_anchor=(1.15, 0.15), fontsize=15, title_fontsize=15, ncol=1)

        plt.grid()
        plt.show()
    
    #### Dynamic Plots ####
    def create_networkactivity_animation(self,
                                     number_of_frames = 20,
                                     frame_duration = 100,
                                     node_size = 800,
                                     figure_size = (20, 15),
                                     file_name = 'network_animation'):
        """
        Creates an animation of network activity over time.

        Parameters:
        df_activity (pd.DataFrame): DataFrame containing the activity data for the network nodes.
        number_of_frames (int, optional): Number of frames in the animation. Default is 20.
        frame_duration (int, optional): Duration of each frame in milliseconds. Default is 100.
        node_size (int, optional): Size of the nodes in the network visualization. Default is 800.
        figure_size (tuple, optional): Size of the figure for the network visualization. Default is (20, 15).
        file_name (str, optional): Name of the file to save the animation. Default is 'network_animation.gif'.

        Returns:
        None
        """

        # Load dataset
        G = self.network
        pos = self.position
        edge_colors = self.edge_colors
        nodesize = node_size
        output_dir = "network_plots/" + file_name
        df_activity = self.timeserie

        # Create the output directory if it does not exist
        if not os.path.exists('network_plots'):
            os.makedirs('network_plots/')
        
        # Create the plot directory, overwriting if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Create multiple plots with networkX
        num_frames = number_of_frames  # Number of frames for the GIF
        images = []

        # Generates frames for the GIF
        print("Generating frames for the GIF...")
        for i in tqdm(range(num_frames)):

            # Ensure the node_size array has the same length as the number of nodes in the graph
            node_sizes = nodesize * np.array([df_activity.loc[i].get(node, 0) for node in G.nodes])

            # Normalize the node values for color mapping
            norm = plt.Normalize(vmin=0, vmax=1)
            node_colors = cm.viridis(norm([df_activity.loc[i].get(node, 0) for node in G.nodes]))

            # Plot the network using the positions and 'name' as node identifiers
            plt.figure(figsize=figure_size)
            options = {
                "node_size": node_sizes,
                "node_color": node_colors,
                "edgecolors": "black",
                "linewidths": 2,
                'arrowstyle':'-'
            }
            nx.draw_networkx(G, pos, **options, with_labels=False)
            nx.draw_networkx_labels(G, pos, 
                                    font_size=12, font_family='sans-serif',
                                    clip_on=False,
                                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha = .5),
                                    #bbox_to_anchor=(0.5, 0.5),
                                    verticalalignment='bottom')
            nx.draw_networkx_edges(G,pos,
                                   arrows = True, arrowstyle="->", arrowsize=15, edge_color=edge_colors, width=2)

            # Create legend for node sizes
            sizes = [nodesize/4, nodesize/4, nodesize*3/4, nodesize]
            for size in sizes:
                plt.scatter([], [], s=size, c='gray', alpha=0.5, label=str(size/800))
            
            # Create a colorbar for the node activity
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), shrink=0.5, aspect=10, anchor=(-0.1, 0.8), location='right')
            cbar.set_label('Node Activity', rotation=270, labelpad=18, fontsize = 15)

            plt.legend(scatterpoints=1, 
                    frameon=True, labelspacing=1, 
                    title='Node Activity', loc='lower right', edgecolor='black', bbox_to_anchor=(1.15, 0.15), fontsize=15, title_fontsize=15, ncol=1)

            plt.grid()
            plt.title(f"Timepoint: {i}")

            # Save the plot as an image file
            filename = f"{output_dir}/frame_{i}.png"
            plt.savefig(filename)
            plt.close()

            # Append the image to the list for creating GIF
            images.append(Image.open(filename))

        # Create and save the GIF
        gif_path = file_name + ".gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=frame_duration,  # Duration in milliseconds between frames
            loop=0  # Loop indefinitely
        )

        # Notify that the process is finished
        print(f"GIF animation saved at {gif_path}")

    def plot_network_activity_comparison(self, activity_mtx1, activity_mtx2, 
                                         timepoint = 0, node_size = 800, figure_size = (20,15),
                                         max_activity = 1, min_activity = -1):
        """
        Plots the network activity at a given timepoint.

        Parameters:
        -----------
        timepoint : int, optional
            The timepoint at which to visualize the network activity. Default is 0.
        node_size : int, optional
            The base size of the nodes in the network. Default is 800.
        figure_size : tuple, optional
            The size of the figure to be created. Default is (20, 15).

        Returns:
        --------
        None
        """
        # Load dataset
        G = self.network
        pos = self.position
        edge_colors = self.edge_colors
        nodesize = node_size

        # Get the different matrix activity
        diff_mtx = activity_mtx1 - activity_mtx2

        # Ensure the node_size array has the same length as the number of nodes in the graph
        diff_mtx_abs = np.abs(diff_mtx)
        node_sizes = nodesize * np.array([diff_mtx_abs.loc[timepoint].get(node, 0) for node in G.nodes])
        
        # Normalize the node values for color mapping
        norm = plt.Normalize(vmin = min_activity, vmax = max_activity)
        node_colors = cm.coolwarm(norm([diff_mtx.loc[timepoint].get(node, 0) for node in G.nodes]))

        # Plot the network using the positions and 'name' as node identifiers
        plt.figure(figsize=figure_size)
        options = {
            "node_size": node_sizes,
            "node_color": node_colors,
            "edgecolors": "black",
            "linewidths": 2,
            'arrowstyle':'-'
        }
        nx.draw_networkx(G, pos, **options, with_labels=False)
        nx.draw_networkx_labels(G, pos, 
                                font_size=12, font_family='sans-serif',
                                clip_on=False,
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha = .5),
                                #bbox_to_anchor=(0.5, 0.5),
                                verticalalignment='bottom')
        nx.draw_networkx_edges(G,pos,
                            arrows = True,
                            arrowstyle="->",
                            arrowsize=15, 
                            edge_color=edge_colors, width=2)

        # Create legend for node sizes
        sizes = [nodesize*1/4, nodesize*1/2, nodesize*3/4, nodesize]
        for size in sizes:
            plt.scatter([], [], s=size, c='gray', alpha=0.5, label=str(size/800))
        
        # Create a colorbar for the node activity
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), shrink=0.5, aspect=10, anchor=(-0.1, 0.8), location='right')
        cbar.set_label('Node Activity', rotation=270, labelpad=18, fontsize = 15)

        plt.legend(scatterpoints=1, 
                frameon=True, labelspacing=1, 
                title='Node Activity', loc='lower right', edgecolor='black', bbox_to_anchor=(1.15, 0.15), fontsize=15, title_fontsize=15, ncol=1)

        plt.grid()
        plt.show()