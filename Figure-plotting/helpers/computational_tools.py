import numpy as np
import math
import xrft
import numpy.fft as npfft
from scipy import signal
import xarray as xr
import os
import gcm_filters

def grid_spacing(param):
        IdxCv = 1. / param.dxCv.values
        IdyCu = 1. / param.dyCu.values
        dyCv = param.dyCv.values
        dxCu = param.dxCu.values
        IareaBu = 1. / (param.dxBu * param.dyBu).values
        return IdxCv, IdyCu, dyCv, dxCu, IareaBu

def velocity(psi, IdxCv, IdyCu, dyCv, dxCu, IareaBu):
    '''
    u = -d psi / dy
    v = d psi / dx
    psi in Q points
    '''
    u = - (psi[1:,:] - psi[:-1,:]) * IdyCu
    v = (psi[:,1:] - psi[:,:-1]) * IdxCv
    return u,v

def vorticity(u, v, IdxCv, IdyCu, dyCv, dxCu, IareaBu):
    vdy = v * dyCv
    dvdx = (vdy[:,1:]-vdy[:,:-1])*IareaBu[:,1:-1]
    
    udy = u * dxCu
    dudy = (udy[1:,:] - udy[:-1])*IareaBu[1:-1,:]
    
    RV = dvdx[1:-1,:] - dudy[:,1:-1]
    return RV
    
def laplace(psi, IdxCv, IdyCu, dyCv, dxCu, IareaBu):
    u, v = velocity(psi, IdxCv, IdyCu, dyCv, dxCu, IareaBu)
    return vorticity(u, v, IdxCv, IdyCu, dyCv, dxCu, IareaBu)
    
def compute_diag(psi, IdxCv, IdyCu, dyCv, dxCu, IareaBu):
    Ny, Nx = psi.shape
    
    D = np.zeros((Ny-2,Nx-2))
    for j in range(1,Ny-1):
        for i in range(1,Nx-1):
            p = 0 * psi
            p[j,i] = 1.
            D[j-1,i-1] = laplace(p, IdxCv, IdyCu, dyCv, dxCu, IareaBu)[j-1,i-1]
    return D

def jacobi_iteration(RV, param, eps=1e-6):
    '''
    Solves the Laplace euqation Laplace(psi) = RV,
    where psi is the streamfunction accounting for the 
    boundary condition (psi=0). RV is the RHS
    which is defined inside the domain, i.e.
    commonly pass ds['R4'].RV.isel(Time=-1,zl=0)[1:-1,1:-1]
    '''
    grid = grid_spacing(param)
    Ny, Nx = RV.shape
    psi = np.zeros((Ny+2,Nx+2))
    D = compute_diag(psi, *grid)
    for i in range(1000000000):
        psi[1:-1,1:-1] = (RV-(laplace(psi, *grid)-D*psi[1:-1,1:-1]))/D
        rel_residual = np.sqrt(((RV - laplace(psi, *grid))**2).mean()) / np.sqrt(((RV)**2).mean())
        if (rel_residual < eps):
            break
    return psi

def optimal_amplitude(ZBx,ZBy,Smagx,Smagy,SGSx,SGSy,u,v,amp_Eng):
    '''
    Model:
    SGSx = Smagx + amp * ZBx
    SGSy = Smagy + amp * ZBy
    '''
    def sel(x):
        return select_LatLon(x).sel(Time=slice(3650,7300))
    ZBx = sel(ZBx)
    ZBy = sel(ZBy)
    Smagx = sel(Smagx)
    Smagy = sel(Smagy)
    SGSx = sel(SGSx)
    SGSy = sel(SGSy)
    u = sel(u)
    v = sel(v)
    
    y1 = SGSx - Smagx
    y2 = SGSy - Smagy
    x1 = ZBx
    x2 = ZBy
    
    dim_u = [d for d in u.dims if d != 'zl']
    dim_v = [d for d in v.dims if d != 'zl']
    
    def scal(y1,x1,y2,x2):
        return (y1*x1).sum(dim=dim_u)+(y2*x2).sum(dim=dim_v)
    
    # MSE optimization
    amp_MSE = scal(y1,x1,y2,x2) / scal(x1,x1,x2,x2)
    # Energy influx optimization
    #amp_Eng = scal(y1,u,y2,v) / scal(x1,u,x2,v)
    
    def fMSE(x,y):
        try:
            return ((x-y)**2).sum(dim=dim_u) / ((y)**2).sum(dim=dim_u)
        except:
            return ((x-y)**2).sum(dim=dim_v) / ((y)**2).sum(dim=dim_v)
    
    # metrics
    MSE     = (fMSE(amp_MSE*ZBx+Smagx,SGSx) + fMSE(amp_MSE*ZBy+Smagy,SGSy))/2
    MSE_Eng = (fMSE(amp_Eng*ZBx+Smagx,SGSx) + fMSE(amp_Eng*ZBy+Smagy,SGSy))/2
    
    corr = (xr.corr(ZBx.chunk({'zl':1,'Time':1}),SGSx,dim=dim_u)+xr.corr(ZBy.chunk({'zl':1,'Time':1}),SGSy,dim=dim_v))/2
    return amp_MSE, MSE, MSE_Eng, corr

def filter_apply(q):
    x = x_coord(q)
    y = y_coord(q)
    def sel(i,j):
        qsel = q.isel({x.name: slice(1+i,-1+i if i<1 else None), y.name: slice(1+j,-1+j if j<1 else None)})
        qsel[x.name] = q[x.name][1:-1]
        qsel[y.name] = q[y.name][1:-1]
        return qsel
    wside = 1. / 8.
    wcorner = 1. / 16.
    wcenter = 1. - (wside*4. + wcorner*4.)

    qf =  wcenter * sel(0,0)   \
        + wcorner * sel(-1,-1) \
        + wcorner * sel(-1,1)  \
        + wcorner * sel(1,-1)  \
        + wcorner * sel(1,1)   \
        + wside   * sel(-1,0)  \
        + wside   * sel(1,0)   \
        + wside   * sel(0,-1)  \
        + wside   * sel(0,1)
    
    return remesh(qf,q)

def filter_AD(q, nwidth=1, norder=0):
    '''
    Implements operator
    sum_{i=0}^{norder} (I - G^nwidth)^i = 
    I + (I-G^nwidth) + (I-G^nwidth)^2 + ... + (I-G^nwidth)^norder
    for norder=0 returns the same field
    '''
    if norder==0 or nwidth==0:
        return q
    
    residual = q
    for i in range(norder):
        residual = I_minus_G(residual, nwidth)
        q = q + residual

    return q

def I_minus_G(q, nwidth=1, mask=1):
    '''
    Implements operator
    I - G^nwidth
    '''
    
    q0 = q
    for j in range(nwidth):
        q = filter_apply(q*mask)*mask

    return q0 - q

def I_minus_G_nselect(q, nwidth=1, nselect=1, mask=1):
    '''
    Implements operator
    (I-G^nwidth)^nselect
    '''
    if nselect==0 or nwidth==0:
        return q
    
    for i in range(nselect):
        q = I_minus_G(q, nwidth, mask)

    return q

def filter_iteration(q, nwidth=0, nselect=1, h=None, residual=False):
    '''
    nwidth - width integer parameter
    nselect - selectivity integer parameter
    Total operator is I - (I-G^nwidth)^nselect
    '''
    if nwidth == 0 or nselect == 0:
        return q
    
    if h is not None:
        h = remesh(h,q)
        mask = h>2e-10
    else:
        mask = 1

    q = q * mask
    
    if residual:
        return I_minus_G_nselect(q, nwidth, nselect, mask)
    else:
        return q - I_minus_G_nselect(q, nwidth, nselect, mask)

def diffy_tu(array,target):
    '''
    finite y-difference of array defined in 
    V points, and result belongs to T (or U) points
    target - T-array to inherit coordinates from
    '''
    p = array.isel(yq=slice(1,None)).rename({'yq':'yh'})
    p['yh'] = target['yh']
    m = array.isel(yq=slice(None,-1)).rename({'yq':'yh'})
    m['yh'] = target['yh']
    return remesh(p-m,target)

def diffx_tv(array,target):
    '''
    finite x-difference of array defined in 
    U points, and result belongs to T (or V) points
    target - T-array to inherit coordinates from
    '''
    p = array.isel(xq=slice(1,None)).rename({'xq':'xh'})
    p['xh'] = target['xh']
    m = array.isel(xq=slice(None,-1)).rename({'xq':'xh'})
    m['xh'] = target['xh']
    return remesh(p-m,target)

def diffx_uq(array,target):
    '''
    finite x-difference of array defined in 
    T points, and result belongs to U (or Q) points
    target - u-array to inherit coordinates from
    '''
    p = array.isel(xh=slice(1,None)).rename({'xh':'xq'})
    p['xq'] = target['xq'].isel(xq=slice(1,-1))
    m = array.isel(xh=slice(None,-1)).rename({'xh':'xq'})
    m['xq'] = target['xq'].isel(xq=slice(1,-1))
    return remesh(p-m,target)

def diffy_vq(array,target):
    '''
    finite y-difference of array defined in 
    T points, and result belongs to V (or Q) points
    target - v-array to inherit coordinates from
    '''
    p = array.isel(yh=slice(1,None)).rename({'yh':'yq'})
    p['yq'] = target['yq'].isel(yq=slice(1,-1))
    m = array.isel(yh=slice(None,-1)).rename({'yh':'yq'})
    m['yq'] = target['yq'].isel(yq=slice(1,-1))
    return remesh(p-m,target)

def prodx_uq(array,target):
    '''
    product in x direction of array defined in 
    T points, and result belongs to U (or Q) points
    target - u-array to inherit coordinates from
    '''
    p = array.isel(xh=slice(1,None)).rename({'xh':'xq'})
    p['xq'] = target['xq'].isel(xq=slice(1,-1))
    m = array.isel(xh=slice(None,-1)).rename({'xh':'xq'})
    m['xq'] = target['xq'].isel(xq=slice(1,-1))
    return remesh(p*m,target)

def prody_vq(array,target):
    '''
    product in y direction of array defined in 
    T points, and result belongs to V (or Q) points
    target - v-array to inherit coordinates from
    '''
    p = array.isel(yh=slice(1,None)).rename({'yh':'yq'})
    p['yq'] = target['yq'].isel(yq=slice(1,-1))
    m = array.isel(yh=slice(None,-1)).rename({'yh':'yq'})
    m['yq'] = target['yq'].isel(yq=slice(1,-1))
    return remesh(p*m,target)

def x_coord(array):
    '''
    Returns horizontal coordinate, 'xq' or 'xh'
    as xarray
    '''
    try:
        coord = array.xq
    except:
        coord = array.xh
    return coord

def y_coord(array):
    '''
    Returns horizontal coordinate, 'yq' or 'yh'
    as xarray
    '''
    try:
        coord = array.yq
    except:
        coord = array.yh
    return coord

def rename_coordinates(xr_dataset):
    '''
    in-place change of coordinate names to Longitude and Latitude.
    For convenience of plotting with xarray.plot()
    '''
    for key in ['xq', 'xh']:
        try:
            xr_dataset[key].attrs['long_name'] = 'Longitude'
            xr_dataset[key].attrs['units'] = ''
        except:
            pass

    for key in ['yq', 'yh']:
        try:
            xr_dataset[key].attrs['long_name'] = 'Latitude'
            xr_dataset[key].attrs['units'] = ''
        except:
            pass

def select_LatLon(array, Lat=(35,45), Lon=(5,15)):
    '''
    array is xarray
    Lat, Lon = tuples of floats
    '''
    x = x_coord(array)
    y = y_coord(array)

    return array.sel({x.name: slice(Lon[0],Lon[1]), 
                      y.name: slice(Lat[0],Lat[1])})

def remesh(input, target, fillna=True):
    '''
    Input and target should be xarrays of any type (u-array, v-array, q-array, h-array).
    Datasets are prohibited.
    Horizontal mesh of input changes according to horizontal mesh of target.
    Other dimensions are unchanged!

    If type of arrays is different:
        - Interpolation to correct points occurs
    If input is Hi-res:
        - Coarsening with integer grain and subsequent interpolation to correct mesh if needed
    if input is Lo-res:
        - Interpolation to Hi-res mesh occurs

    Input and output Nan values are treates as zeros (see "fillna")
    '''

    # Define coordinates
    x_input  = x_coord(input)
    y_input  = y_coord(input)
    x_target = x_coord(target)
    y_target = y_coord(target)

    # ratio of mesh steps
    ratiox = np.diff(x_target)[0] / np.diff(x_input)[0]
    ratiox = math.ceil(ratiox)

    ratioy = np.diff(y_target)[0] / np.diff(y_input)[0]
    ratioy = math.ceil(ratioy)
    
    # B.C.
    if fillna:
        result = input.fillna(0)
    else:
        result = input
    
    if (ratiox > 1 or ratioy > 1):
        # Coarsening; x_input.name returns 'xq' or 'xh'
        result = result.coarsen({x_input.name: ratiox, y_input.name: ratioy}, boundary='pad').mean()

    # Coordinate points could change after coarsegraining
    x_result = x_coord(result)
    y_result = y_coord(result)

    # Interpolate if needed
    if not x_result.equals(x_target) or not y_result.equals(y_target):
        result = result.interp({x_result.name: x_target, y_result.name: y_target})
        if fillna:
            result = result.fillna(0)

    # Remove unnecessary coordinates
    if x_target.name != x_input.name:
        result = result.drop_vars(x_input.name)
    if y_target.name != y_input.name:
        result = result.drop_vars(y_input.name)
    
    return result

def gaussian_remesh(_input, target, FGR=2):
    '''
    input - xr.DataArray() on high-resolution grid
    target - any xr.DataArray() containing target coordinates;
    returns filtered and coarsegrained version of input_field
    '''
    # Define grid ratio
    ratio = math.ceil(np.diff(x_coord(target))[0] / np.diff(x_coord(_input))[0])

    G = gcm_filters.Filter(filter_scale = ratio * FGR, dx_min=1) 
    
    # Find spatial coordinates
    x = 'xh' if 'xh' in _input.dims else 'xq'
    y = 'yh' if 'yh' in _input.dims else 'yq'    
    
    filtered = G.apply(_input, dims=(y,x))
    
    # Coarsegrain
    coarsegrained = remesh(filtered, target)
    
    return coarsegrained

def compute_isotropic_KE(u_in, v_in, dx, dy, Lat=(35,45), Lon=(5,15), window='hann', 
        nfactor=2, truncate=True, detrend='linear', window_correction=True, nd_wavenumber=False):
    '''
    u, v - "velocity" arrays defined on corresponding staggered grids
    dx, dy - grid step arrays defined in the center of the cells
    Default options: window correction + linear detrending
    Output:
    mean(u^2+v^2)/2 = int(E(k),dk)
    This equality is expected for detrend=None, window='boxcar'
    freq_r - radial wavenumber, m^-1
    window = 'boxcar' or 'hann'
    '''
    # Interpolate to the center of the cells
    u = remesh(u_in, dx)
    v = remesh(v_in, dy)

    # Select desired Lon-Lat square
    u = select_LatLon(u,Lat,Lon)
    v = select_LatLon(v,Lat,Lon)

    # mean grid spacing in metres
    dx = select_LatLon(dx,Lat,Lon).mean().values
    dy = select_LatLon(dy,Lat,Lon).mean().values

    # define uniform grid
    x = dx*np.arange(len(u.xh))
    y = dy*np.arange(len(u.yh))
    u['xh'] = x
    u['yh'] = y
    v['xh'] = x
    v['yh'] = y

    Eu = xrft.isotropic_power_spectrum(u, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)
    Ev = xrft.isotropic_power_spectrum(v, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = (Eu+Ev) / 2 # because power spectrum is twice the energy
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers

    if nd_wavenumber:
        Lx = x.max() - x.min()
        Ly = y.max() - y.min()
        kmin = 2*np.pi * min(1/Lx, 1/Ly)
        E['freq_r'] = E['freq_r'] / kmin
        E = E * kmin
    
    ############## normalization tester #############
    #print('Energy balance:')
    #print('mean(u^2+v^2)/2=', ((u**2+v**2)/2).mean(dim=('Time', 'xh', 'yh')).values)
    #spacing = np.diff(E.freq_r).mean()
    #print('int(E(k),dk)=', (E.sum(dim='freq_r').mean(dim='Time') * spacing).values)
    #print(f'Max wavenumber={E.freq_r.max().values} [1/m], \n x-grid-scale={np.pi/dx} [1/m], \n y-grid-scale={np.pi/dy} [1/m]')
    
    return E

def compute_isotropic_cospectrum(u_in, v_in, fu_in, fv_in, dx, dy, Lat=(35,45), Lon=(5,15), window='hann', 
        nfactor=2, truncate=False, detrend='linear', window_correction=True):
    # Interpolate to the center of the cells
    u = remesh(u_in, dx)
    v = remesh(v_in, dy)
    fu = remesh(fu_in, dx).transpose(*u.dims)
    fv = remesh(fv_in, dy).transpose(*v.dims)

    # Select desired Lon-Lat square
    u = select_LatLon(u,Lat,Lon)
    v = select_LatLon(v,Lat,Lon)
    fu = select_LatLon(fu,Lat,Lon)
    fv = select_LatLon(fv,Lat,Lon)

    # mean grid spacing in metres
    dx = select_LatLon(dx,Lat,Lon).mean().values
    dy = select_LatLon(dy,Lat,Lon).mean().values

    # define uniform grid
    x = dx*np.arange(len(u.xh))
    y = dy*np.arange(len(u.yh))
    for variable in [u, v, fu, fv]:
        variable['xh'] = x
        variable['yh'] = y

    Eu = xrft.isotropic_cross_spectrum(u, fu, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)
    Ev = xrft.isotropic_cross_spectrum(v, fv, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = (Eu+Ev)
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers
    
    return np.real(E)

def compute_isotropic_PE(h_int, dx, dy, Lat=(35,45), Lon=(5,15), window='hann', 
        nfactor=2, truncate=True, detrend='linear', window_correction=True):
    '''
    hint - interface displacement in metres
    dx, dy - grid step arrays defined in the center of the cells
    Default options: window correction + linear detrending
    Output:
    mean(h^2)/2 = int(E(k),dk)
    This equality is expected for detrend=None, window='boxcar'
    freq_r - radial wavenumber, m^-1
    window = 'boxcar' or 'hann'
    '''
    # Select desired Lon-Lat square
    hint = select_LatLon(h_int,Lat,Lon)

    # mean grid spacing in metres
    dx = select_LatLon(dx,Lat,Lon).mean().values
    dy = select_LatLon(dy,Lat,Lon).mean().values

    # define uniform grid
    x = dx*np.arange(len(hint.xh))
    y = dy*np.arange(len(hint.yh))
    hint['xh'] = x
    hint['yh'] = y

    E = xrft.isotropic_power_spectrum(hint, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = E / 2 # because power spectrum is twice the energy
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers
    
    return E

def compute_KE_time_spectrum(u_in, v_in, Lat=(35,45), Lon=(5,15), Time=slice(0,None), window='hann', 
        nchunks=2, detrend='linear', window_correction=True):
    '''
    Returns KE spectrum with normalization:
    mean(u^2+v^2)/2 = int(E(nu),dnu),
    where nu - time frequency in 1/day (not "angle frequency")
    E(nu) - energy density, i.e. m^2/s^2 * day
    '''

    # Select range of Lat-Lon-time
    u = select_LatLon(u_in,Lat,Lon).sel(Time=Time)
    v = select_LatLon(v_in,Lat,Lon).sel(Time=Time)

    # Let integer division by nchunks
    nTime = len(u.Time)
    chunk_length = math.floor(nTime / nchunks)
    nTime = chunk_length * nchunks

    # Divide time series to time chunks
    u = u.isel(Time=slice(nTime)).chunk({'Time': chunk_length})
    v = v.isel(Time=slice(nTime)).chunk({'Time': chunk_length})

    # compute spatial-average time spectrum
    ps_u = xrft.power_spectrum(u, dim='Time', window=window, window_correction=window_correction, 
        detrend=detrend, chunks_to_segments=True).mean(dim=('xq','yh'))
    ps_v = xrft.power_spectrum(v, dim='Time', window=window, window_correction=window_correction, 
        detrend=detrend, chunks_to_segments=True).mean(dim=('xh','yq'))

    ps = ps_u + ps_v

    # in case of nchunks > 1
    try:
        ps = ps.mean(dim='Time_segment')
    except:
        pass

    # Convert 2-sided power spectrum to one-sided
    ps = ps[ps.freq_Time>=0]
    freq = ps.freq_Time
    ps[freq==0] = ps[freq==0] / 2

    # Drop zero frequency for better plotting
    ps = ps[ps.freq_Time>0]

    ############## normalization tester #############
    #print('Energy balance:')
    #print('mean(u^2+v^2)/2=', ((u**2)/2).mean(dim=('Time', 'xq', 'yh')).values + ((v**2)/2).mean(dim=('Time', 'xh', 'yq')).values)
    #print('int(E(nu),dnu)=', (ps.sum(dim='freq_Time') * ps.freq_Time.spacing).values)
    #spacing = np.diff(u.Time).mean()
    #print(f'Minimum period {2*spacing} [days]')
    #print(f'Max frequency={ps.freq_Time.max().values} [1/day], \n Max inverse period={0.5/spacing} [1/day]')

    return ps

def mass_average(KE, h, dx, dy):
    return (KE*h*dx*dy).mean(dim=('xh', 'yh')) / (h*dx*dy).mean(dim=('xh', 'yh'))

def Lk_error(input, target, normalize=False, k=2):
    '''
    Universal function for computation of NORMALIZED error.
    target - "good simulation", it is used for normalization
    Output is a scalar value.
    error = target-input
    result = mean(abs(error)) / mean(abs(target))
    numerator and denominator could be vectors
    only if variables have layers.
    In this case list of two elements is returned
    '''
    # Check dimensions
    if sorted(input.dims) != sorted(target.dims) or sorted(input.shape) != sorted(target.shape):
        import sys
        sys.exit(f'Dimensions disagree: {sorted(input.dims)} {sorted(target.dims)} {sorted(input.shape)} {sorted(target.shape)}')

    error = target - input

    average_dims = list(input.dims)
    
    # if layer is present, do not average over it at first stage!
    
    if 'zl' in average_dims:
        average_dims.remove('zl')

    if 'zi' in average_dims:
        average_dims.remove('zi')
    
    def lk_norm(x,k):
        '''
        k - order of norm
        k = -1 is the L-infinity norm
        '''
        if k > 0:
            return ((np.abs(x)**k).mean(dim=average_dims))**(1./k)
        elif k==-1:
            return np.abs(x).max(dim=average_dims)
    
    result = lk_norm(error,k)
    if normalize:
        result = result / lk_norm(target,k)

    return list(np.atleast_1d(result))