import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import imageio

def create_animation(fun, idx, filename='my-animation.gif', dpi=200, FPS=18, loop=0, deezering=True):
    '''
    See https://pythonprogramming.altervista.org/png-to-gif/
    fun(i) - a function creating one snapshot, has only one input:
        - number of frame i
    idx - range of frames, i in idx
    FPS - frames per second
    filename - animation name
    dpi - set 300 or so to increase quality
    loop - number of repeats of the gif
    '''
    frames = []
    for i in idx:
        fun(i)
        plt.savefig('.frame.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        if deezering:
            frames.append(Image.open('.frame.png').convert('RGB'))
        else:
            frames.append(Image.open('.frame.png'))
        print(f'Frame {i} is created', end='\r')
    os.system('rm .frame.png')
    # How long to persist one frame in milliseconds to have a desired FPS
    duration = 1000 / FPS
    print(f'Animation at FPS={FPS} will last for {len(idx)/FPS} seconds')
    frames[0].save(
        filename, format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop)
    
def merge_gifs(gif_files, output_file, fps=20):
    '''
    Note it is purely chatgpt code
    '''
    # Get a list of all GIF files in the input folder

    # Create a list to store individual frames
    frames = []

    # Read each GIF file and extract frames
    for gif_file in gif_files:
        gif_path = os.path.join(gif_file)
        try:
            with imageio.get_reader(gif_path) as reader:
                for frame in reader:
                    frames.append(frame)
        except Exception as e:
            print(f"Error reading {gif_file}: {e}")

    # Write the merged frames to the output GIF
    try:
        with imageio.get_writer(output_file, mode='I', duration=1000//fps, loop=0) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Merged {len(gif_files)} GIFs into {output_file}")
    except Exception as e:
        print(f"Error writing {output_file}: {e}")

def split_gif(input_file, output_folder, n):
    '''
    Note it is purely chatgpt code
    '''
    try:
        with imageio.get_reader(input_file) as reader:
            num_frames = len(reader)
            frames_per_segment = num_frames // n

            if frames_per_segment == 0:
                print("Cannot split into that many segments. Try a smaller value of n.")
                return

            for i in range(n):
                start_frame = i * frames_per_segment
                end_frame = (i + 1) * frames_per_segment if i < n - 1 else num_frames

                os.system('mkdir -p ' + output_folder)
                output_file = os.path.join(output_folder, f"segment_{i}.gif")

                with imageio.get_writer(output_file, mode='I', duration=reader.get_meta_data()['duration'], loop=0) as writer:
                    for frame_number in range(start_frame, end_frame):
                        frame = reader.get_data(frame_number)
                        writer.append_data(frame)

                print(f"Segment {i} saved as {output_file}")

        print(f"Split {input_file} into {n} segments")
    except Exception as e:
        print(f"Error splitting {input_file}: {e}")
    
def default_rcParams(kw={}):
    '''
    Also matplotlib.rcParamsDefault contains the default values,
    but:
    - backend is changed
    - without plotting something as initialization,
    inline does not work
    '''
    plt.plot()
    plt.close()
    rcParams = matplotlib.rcParamsDefault.copy()
    
    # We do not change backend because it can break
    # inlining; Also, 'backend' key is broken and 
    # we cannot use pop method
    for key, val in rcParams.items():
        if key != 'backend':
            rcParams[key] = val

    matplotlib.rcParams.update({
        'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm',

        'figure.figsize': (4, 4),

        'figure.subplot.wspace': 0.3,
        
        'font.size': 14,
        #'axes.labelsize': 10,
        #'axes.titlesize': 12,
        #'xtick.labelsize': 10,
        #'ytick.labelsize': 10,
        #'legend.fontsize': 10,

        'axes.formatter.limits': (-2,3),
        'axes.formatter.use_mathtext': True,
        'axes.labelpad': 0,
        'axes.titlelocation' : 'center',
        
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    matplotlib.rcParams.update(**kw)

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str
    
def imshow(_q, cbar=True, location='right', cbar_label=None, ax=None, cmap=None, 
    vmax = None, vmin = None, pct=99, axes=False, interpolation='none', normalize='False', normalize_postfix='', **kwargs):

    def rms(x):
        return float(np.sqrt(np.mean(x.astype('float64')**2)))
    def mean(x):
        return float(np.mean(x.astype('float64')))

    if normalize != 'False':
        if normalize == 'mean':
            q_norm = mean(_q)
            q_str = f'$\\mu_x={latex_float(q_norm)}$'
        else:
            q_norm = rms(_q)
            q_str = f'${latex_float(q_norm)}$'    
        q = _q / q_norm
        if len(normalize_postfix) > 0:
            q_str += f' {normalize_postfix}'
    else:
        q = _q

    if q.min() < 0:
        vmax = np.percentile(np.abs(q), pct) if vmax is None else vmax
        vmin = -vmax if vmin is None else vmin
    else:
        vmax = np.percentile(q, pct) if vmax is None else vmax
        vmin = 0 if vmin is None else vmin

    cmap=cmocean.cm.balance if cmap is None else cmap
    
    kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    
    if ax is None:
        ax = plt.gca()

    # flipud because imshow inverts vertical axis
    im = ax.imshow(np.flipud(q), **kw, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    if normalize != 'False':
        ax.text(0.05,0.85,q_str,transform = ax.transAxes, fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    if cbar:
        divider = make_axes_locatable(ax)
        if location == 'right':
            cax = divider.append_axes('right', size="5%", pad=0.1)
            cbar_kw = dict()
        elif location == 'bottom':
            cax = divider.append_axes('bottom', size="5%", pad=0.1)
            cbar_kw = dict(orientation='horizontal')
        cb = plt.colorbar(im, cax = cax, label=cbar_label, **cbar_kw)
    
    # Return axis to initial image
    plt.sca(ax)
    return im

def set_letters(x=-0.2, y=1.05, fontsize=11, letters=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], color='k'):
    fig = plt.gcf()
    axes = fig.axes
    j = 0
    for ax in axes:
        if hasattr(ax, 'collections'):
            if len(ax.collections) > 0:
                collection = ax.collections[0]
            else:
                collection = ax.collections
            if isinstance(collection, matplotlib.collections.LineCollection):
                print('Colorbar-like object skipped')
            else:
                try:
                    ax.text(x,y,f'({letters[j]})', transform = ax.transAxes, fontweight='bold', fontsize=fontsize, color=color)
                except:
                    print('Cannot set letter', letters[j])
                j += 1
        