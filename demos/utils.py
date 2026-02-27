# Copyright (C) 2026 Antonio Baiano Svizzero and Jørgen S. Dokken
#
# This file is part of FEniCSx_PML (https://github.com/bayswiss/fenicsx_pml)
#
# SPDX-License-Identifier:    GPL-3.0-or-later

import numpy as np
import numpy.typing as npt 
from dolfinx import geometry, default_scalar_type
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MicrophonePressure:
    def __init__(self, domain, microphone_position):
        """Initialize microphone(s).

        Args:
            domain: The domain to insert microphones on
            microphone_position: Position of the microphone(s).
                Assumed to be ordered as ``(mic0_x, mic1_x, ..., mic0_y, mic1_y, ..., mic0_z, mic1_z, ...)``

        """
        self._domain = domain
        self._position = np.asarray(
            microphone_position, dtype=domain.geometry.x.dtype
        ).reshape(3, -1)
        self._local_cells, self._local_position = self.compute_local_microphones()

    def compute_local_microphones(
        self,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]]:
        """
        Compute the local microphone positions for a distributed mesh

        Returns:
            Two lists (local_cells, local_points) containing the local cell indices and the local points
        """
        points = self._position.T
        bb_tree = geometry.bb_tree(self._domain, self._domain.topology.dim)

        cells = []
        points_on_proc = []

        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(
            self._domain, cell_candidates, points
        )

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        return np.asarray(cells, dtype=np.int32), np.asarray(
            points_on_proc, dtype=self._domain.geometry.x.dtype
        )

    def listen(
        self, 
        uh,
        recompute_collisions: bool = False
    ) -> npt.NDArray[np.complexfloating]:
        if recompute_collisions:
            self._local_cells, self._local_position = self.compute_local_microphones()
        if len(self._local_cells) > 0:
            return uh.eval(self._local_position, self._local_cells)
        else:
            return np.zeros(0, dtype=default_scalar_type)

def plot_complex_spectra(x_axis, p_spectra_list, labels=None, title=None, plot_db=False):
    """
    Plots the amplitude and phase of one OR MULTIPLE complex arrays.
    
    Arguments:
    - p_complex_list (list or numpy.ndarray): A single 1D complex array, a list of 
                                              1D complex arrays, or a 2D array.
    - x_axis (numpy.ndarray, optional): The x-axis values. Defaults to indices.
    - labels (list of str, optional): Names for the legend.
    - title (str, optional): The title for the top plot.
    - plot_db (bool, optional): If True, plots amplitude in decibels (dB). Defaults to False.
    """
    
    # 1. Standardize input to always be a list of arrays
    if isinstance(p_spectra_list, np.ndarray) and p_spectra_list.ndim == 1:
        p_spectra_list = [p_spectra_list] # Wrap single array in a list
    elif isinstance(p_spectra_list, np.ndarray) and p_spectra_list.ndim == 2:
        p_spectra_list = list(p_spectra_list) # Convert 2D array rows to list

    # Set default x_axis if not provided
    if x_axis is None:
        x_axis = np.arange(len(p_spectra_list[0]))
        x_label = 'Index'
    else:
        x_label = 'Frequency / X-axis'

    # 2. Set up the Figure and GridSpec
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    ax_amp = fig.add_subplot(gs[0:2, :])
    ax_phase = fig.add_subplot(gs[2, :])

    # 3. Loop through the complex arrays and plot them
    for i, p_complex in enumerate(p_spectra_list):
        amplitude = np.abs(p_complex)
        
        # NEW: Convert to dB if the flag is True (using 1e-12 to prevent log(0) warnings)
        if plot_db:
            amplitude = 20 * np.log10(np.maximum(amplitude, 1e-12))
            
        phase = np.angle(p_complex)
        
        # Figure out the label for the legend
        if labels is not None and i < len(labels):
            current_label = labels[i]
        else:
            current_label = f'Signal {i+1}'

        # Plot Amplitude
        ax_amp.plot(x_axis, amplitude, linewidth=1.5, label=current_label)
        
        # Plot Phase
        ax_phase.plot(x_axis, phase, linewidth=1.5, label=current_label, alpha=0.8)

    # 4. Format the Amplitude Plot
    ax_amp.set_title(title, fontsize=14, fontweight='bold')
    
    # NEW: Dynamically update the y-axis label
    amp_ylabel = 'Amplitude (dB)' if plot_db else 'Amplitude (Linear)'
    ax_amp.set_ylabel(amp_ylabel, fontsize=12)
    
    ax_amp.grid(True, linestyle='--', alpha=0.7)
    ax_amp.set_xlim([min(x_axis), max(x_axis)])
    ax_amp.tick_params(labelbottom=False)
    ax_amp.legend() # Add legend here

    # 5. Format the Phase Plot
    ax_phase.set_xlabel(x_label, fontsize=12)
    ax_phase.set_ylabel('Phase (Rad)', fontsize=12)
    ax_phase.grid(True, linestyle='--', alpha=0.7)
    ax_phase.set_xlim([min(x_axis), max(x_axis)])
    ax_phase.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_phase.set_yticklabels(['-$\pi$', '-$\pi/2$', '0', '$\pi/2$', '$\pi$'])

    plt.tight_layout()
    plt.show()
    
    return fig, (ax_amp, ax_phase)