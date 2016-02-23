'''
Functions for plotting surfaces with pure python code. 
Reduced version of nilearn surface plotting:
https://github.com/juhuntenburg/nilearn/tree/enh/surface_plotting

Helper function for symmetric colormap is copied from nilearn.
'''

def plot_surf_stat_map(coords, faces, stat_map=None,
                       elev=0, azim=0,
                       cmap='coolwarm',
                       threshold=None, bg_map=None,
                       bg_on_stat=False,
                       alpha='auto',
                       vmin=None, vmax=None,
                       cbar='sequential', # or'diverging'
                       symmetric_cbar="auto",
                       figsize=None,
                       labels=None, label_col=None, label_cpal=None,
                       mask=None, mask_lenient=None,
                       **kwargs):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    
    # load mesh and derive axes limits
    faces = np.array(faces, dtype=int)
    limits = [coords.min(), coords.max()]

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if cmap is given as string, translate to matplotlib cmap
    if type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap)

    # initiate figure and 3d axes
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', xlim=limits, ylim=limits)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # plot mesh without data
    p3dcollec = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                triangles=faces, linewidth=0.,
                                antialiased=False,
                                color='white')

    # where mask is indices of nodes to include:
    if mask is not None:    
        cmask = np.zeros(len(coords))
        cmask[mask] = 1
        cutoff = 2 # include triangles in cortex only if ALL nodes in mask
        if mask_lenient: # include triangles in cortex if ANY are in mask
            cutoff = 0
        fmask = np.where(cmask[faces].sum(axis=1) > cutoff)[0]    
    
    
    # If depth_map and/or stat_map are provided, map these onto the surface
    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    if bg_map is not None or stat_map is not None:

        face_colors = np.ones((faces.shape[0], 4))
        face_colors[:, :3] = .5*face_colors[:, :3]

        if bg_map is not None:
            bg_data = bg_map
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError('The bg_map does not have the same number '
                                 'of vertices as the mesh.')
            bg_faces = np.mean(bg_data[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()
            face_colors = plt.cm.gray_r(bg_faces)

        # modify alpha values of background
        face_colors[:, 3] = alpha*face_colors[:, 3]

        if stat_map is not None:
            stat_map_data = stat_map
            stat_map_faces = np.mean(stat_map_data[faces], axis=1)

            if cbar is 'diverging':
                print cbar
                # Call _get_plot_stat_map_params to derive symmetric vmin and vmax
                # And colorbar limits depending on symmetric_cbar settings
                cbar_vmin, cbar_vmax, vmin, vmax = \
                    _get_plot_stat_map_params(stat_map_faces, vmax,
                                              symmetric_cbar, kwargs)
            if cbar is 'sequential':
                if vmin is None:
                    vmin = stat_map_data.min()
                if vmax is None:
                    vmax = stat_map_data.max()

            if threshold is not None:
                kept_indices = np.where(abs(stat_map_faces) >= threshold)[0]
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices]) * face_colors[kept_indices]
                else:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices])
            else:
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    if mask is not None:
                        face_colors[fmask] = cmap(stat_map_faces)[fmask] * face_colors[fmask]
                    else:
                        face_colors = cmap(stat_map_faces) * face_colors
                else:
                    if mask is not None:
                        face_colors[fmask] = cmap(stat_map_faces)[fmask] 
                        
                    else:
                        face_colors = cmap(stat_map_faces)

        if labels is not None:
            '''
            labels requires a tuple of label/s, each a list/array of node indices
            ----------------------------------------------------------------------
            color palette for labels
            if label_cpal is None, outlines will be black
            if it's a color palette name, a different color for each label will be generated
            if it's a list of rgb or color names, these will be used
            valid color names from http://xkcd.com/color/rgb/
            '''
            if label_cpal is not None:
                if label_col is not None:
                    raise ValueError("Don't use label_cpal and label_col together.")         
                if type(label_cpal) == str:
                    cpal = sns.color_palette(label_cpal, len(labels))
                if type(label_cpal) == list:
                    if len(label_cpal) < len(labels):
                        raise ValueError('There are not enough colors in the color list.')
                    try:
                        cpal = sns.color_palette(label_cpal)
                    except:
                        cpal = sns.xkcd_palette(label_cpal)
        
            
                
            
            for n_label, label in enumerate(labels):
                for n_face, face in enumerate(faces):
                    count = len(set(face).intersection(set(label)))
                    if (count > 0) & (count < 3):
                        if label_cpal is None:
                            if label_col is not None:
                                face_colors[n_face,0:3] = sns.xkcd_palette([label_col])[0]
                            else:
                                face_colors[n_face,0:3] = sns.xkcd_palette(["black"])[0]
                        else:
                            face_colors[n_face,0:3] = cpal[n_label]
                            
        p3dcollec.set_facecolors(face_colors)

    return fig


def _get_plot_stat_map_params(stat_map_data, vmax, symmetric_cbar, kwargs,
    force_min_stat_map_value=None):
    import numpy as np
    """ Internal function for setting value limits for plot_stat_map and
    plot_glass_brain.
    The limits for the colormap will always be set to range from -vmax to vmax.
    The limits for the colorbar depend on the symmetric_cbar argument, please
    refer to docstring of plot_stat_map.
    """
    # make sure that the color range is symmetrical
    if vmax is None or symmetric_cbar in ['auto', False]:
        # Avoid dealing with masked_array:
        if hasattr(stat_map_data, '_mask'):
            stat_map_data = np.asarray(
                    stat_map_data[np.logical_not(stat_map_data._mask)])
        stat_map_max = np.nanmax(stat_map_data)
        if force_min_stat_map_value == None:
            stat_map_min = np.nanmin(stat_map_data)
        else:
            stat_map_min = force_min_stat_map_value
    if symmetric_cbar == 'auto':
        symmetric_cbar = stat_map_min < 0 and stat_map_max > 0
    if vmax is None:
        vmax = max(-stat_map_min, stat_map_max)
    if 'vmin' in kwargs:
        raise ValueError('this function does not accept a "vmin" '
            'argument, as it uses a symmetrical range '
            'defined via the vmax argument. To threshold '
            'the map, use the "threshold" argument')
    vmin = -vmax
    if not symmetric_cbar:
        negative_range = stat_map_max <= 0
        positive_range = stat_map_min >= 0
        if positive_range:
            cbar_vmin = 0
            cbar_vmax = None
        elif negative_range:
            cbar_vmax = 0
            cbar_vmin = None
        else:
            cbar_vmin = stat_map_min
            cbar_vmax = stat_map_max
    else:
        cbar_vmin, cbar_vmax = None, None
    return cbar_vmin, cbar_vmax, vmin, vmax
    

def plot_surf_label(coords, faces, 
                    labels=None,
                    elev=0, azim=0,
                    cpal='bright',
                    threshold=None, 
                    bg_map=None,
                    bg_on_labels=False,
                    alpha='auto',
                    figsize=None,
                    **kwargs):
    
    '''
    - labels requires a tuple of label/s, each a list/array of node indices
    - cpal takes either the name of a seaborn color palette or matplotlib color map, 
      or a list of rgb values or color names from http://xkcd.com/color/rgb/
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    
    # load mesh and derive axes limits
    faces = np.array(faces, dtype=int)
    limits = [coords.min(), coords.max()]

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if cap is given as string, translate to seaborn color palette
    if type(cpal) == str:
        cpal = sns.color_palette(cpal, len(labels))
    if type(cpal) == list:
        if len(cpal) < len(labels):
            raise ValueError('There are not enough colors in the color list.')
        try:
            cpal = sns.color_palette(cpal)
        except:
            cpal = sns.xkcd_palette(cpal)
    
    # initiate figure and 3d axes
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', xlim=limits, ylim=limits)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # plot mesh without data
    p3dcollec = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                triangles=faces, linewidth=0.,
                                antialiased=False,
                                color='white')

    if bg_map is not None or labels is not None:

        face_colors = np.ones((faces.shape[0], 4))
        face_colors[:, :3] = .5*face_colors[:, :3]

        if bg_map is not None:
            bg_data = bg_map
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError('The bg_map does not have the same number '
                                 'of vertices as the mesh.')
            bg_faces = np.mean(bg_data[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()    
            face_colors = plt.cm.gray_r(bg_faces)

        # modify alpha values of background
        face_colors[:, 3] = alpha*face_colors[:, 3]

        # color the labels, either overriding or overlaying bg_map
        if labels is not None:
            for n_label,label in enumerate(labels):
                for n_face, face in enumerate(faces):
                    count = len(set(face).intersection(set(label)))
                    if count > 1:
                        if bg_on_labels:
                            face_colors[n_face,0:3] = cpal[n_label] * face_colors[n_face,0:3]
                        else:
                            face_colors[n_face,0:3] = cpal[n_label]
            
        p3dcollec.set_facecolors(face_colors)

    return fig

    
def crop_img(fig, margin=10):
    # takes fig, returns image
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import os
    
    fig.savefig('./tempimage', bbox_inches='tight', orientation='landscape')
    plt.close(fig)
    img = mpimg.imread('./tempimage.png')
    os.remove('./tempimage.png')    
    
    kept = {'rows':[], 'cols':[]}
    for row in range(img.shape[0]):
        if len(set(np.ndarray.flatten(img[row,:,:]))) > 1:
            kept['rows'].append(row)
    for col in range(img.shape[1]):
        if len(set(np.ndarray.flatten(img[:,col,:]))) > 1:
            kept['cols'].append(col)
    
    if margin:
        return img[min(kept['rows'])-margin:max(kept['rows'])+margin,
                   min(kept['cols'])-margin:max(kept['cols'])+margin]
    else:
        return img[kept['rows']][:,kept['cols']]
        
        


def create_fig(data=None, labels=None, label_col=None, 
               hemi=None, surf='pial',
               sulc=True, alpha='auto',
               cmap='cubehelix', cpal='bright', cbar=False,
               dmin=None, dmax=None,
               mask=None):
    
    import nibabel as nib, numpy as np
    import matplotlib.pyplot as plt, matplotlib as mpl
    from IPython.core.display import Image, display
    import os     
    
    fsDir = '/afs/cbs.mpg.de/projects/mar004_lsd-lemon-preproc/freesurfer'
    surf_f = '%s/fsaverage5/surf/%s.%s' % (fsDir, hemi, surf)
    coords = nib.freesurfer.io.read_geometry(surf_f)[0]
    faces = nib.freesurfer.io.read_geometry(surf_f)[1]
    if sulc:
        sulc_f = '%s/fsaverage5/surf/%s.sulc' % (fsDir, hemi)
        sulc = nib.freesurfer.io.read_morph_data(sulc_f)
        sulc_bool = True
    else:
        sulc = None
        sulc_bool = False
      
    # create images
    imgs = []        
    for azim in [0, 180]:
        
        if data is not None:
            if dmin is None:
                dmin = data[np.nonzero(data)].min()
            if dmax is None:
                dmax = data.max()
            fig = plot_surf_stat_map(coords, faces, stat_map=data,
                                 elev=0, azim=azim,
                                 cmap=cmap,
                                 bg_map=sulc,bg_on_stat=sulc_bool,
                                 vmin=dmin, vmax=dmax,
                                 labels=labels, label_col=label_col,
                                 alpha=alpha,
                                 mask=mask, mask_lenient=False)
                                 #label_cpal=cpal)
        else:
            fig = plot_surf_label(coords, faces, 
                                  labels=labels,
                                  elev=0, azim=azim,                             
                                  bg_map=sulc,
                                  cpal=cpal,
                                  bg_on_labels=sulc_bool,
                                  alpha=alpha)

        # crop image
        imgs.append((crop_img(fig, margin=15)),)
        plt.close(fig)

    # create figure with color bar
    fig = plt.figure()
    fig.set_size_inches(8, 4)

    ax1 = plt.subplot2grid((4,60), (0,0),  colspan = 26, rowspan =4)
    plt.imshow(imgs[0])
    ax1.set_axis_off()

    ax2 = plt.subplot2grid((4,60), (0,28),  colspan = 26, rowspan =4)
    plt.imshow(imgs[1])
    ax2.set_axis_off()
    
    if cbar==True and data is not None:  
        cax = plt.subplot2grid((4,60), (1,59),  colspan = 1, rowspan =2)
        cmap = plt.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=dmin, vmax=dmax)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([dmin, dmax])
    
    fig.savefig('./tempimage')
    plt.close(fig)
    display(Image(filename='./tempimage.png', width=800))
    os.remove('./tempimage.png')           
