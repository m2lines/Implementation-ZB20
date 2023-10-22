import os
import xarray as xr

class netcdf_property:
    '''
    Implements storage of statistical characteristic
    in experiment folder at netcdf file
    having name KE_key.nc,
    where key - additional usually name of the experiment
    Information about folder in decorated class. See:
        instance.folder
        instance.key
    '''
    def __init__(self, function):
        self.function = function
    
    def __get__(self, instance, owner):
        '''
        Method __get__ is called, when this class(netcdf_cache)
        is accessed as attribute of abother once class (owner),
        or its instance (instance)
        https://python-reference.readthedocs.io/en/latest/docs/dunderdsc/get.html
        '''        
        if instance is None: return self # see https://gist.github.com/asross/952fa456f8bcd07abf684cc515d49030

        funcname = self.function.__name__
        #filename = os.path.join(instance.folder, funcname+'_'+instance.key+'.nc')
        #filename = os.path.join('/scratch/pp2681/mom6/cache', funcname+'_'+instance.key+'.nc')
        filename = os.path.join('/scratch/pp2681/mom6/cache', '-'.join(instance.folder.split('/')[4:-2])+'-'+instance.key+'-'+funcname+'.nc')
        #print(filename)
        if instance.recompute:
            try:
                os.remove(filename)
                #print(f'Removing cache file {filename}')
            except:
                pass

        # Try to open netcdf if exists
        if os.path.exists(filename):
            #print(f'Reading file {filename}')
            try:
                ncfile = xr.open_dataset(filename, decode_times=False, chunks={'Time': 1, 'zl': 1})
            except:
                ncfile = xr.open_dataset(filename, decode_times=False) # for very small files
            #print(f'Returning cached value of {funcname}')
            if funcname in ncfile:
                value = ncfile[funcname]
                ncfile.close() # to prevent out of memory
                if free_of_NaNs_and_zeros(value):
                    return value
                else:
                    os.remove(filename) # value will be recalculated below
            else:
                os.remove(filename) # value will be recalculated below

        #print(f'Calculating value of {funcname}')
        try:
            value = self.function(instance).chunk({'Time':1,'zl':1})
        except:
            value = self.function(instance) # for very small objects
        
        # Save value only if it is not trivial
        if free_of_NaNs_and_zeros(value):
            # Create new dataset
            ncfile = xr.Dataset()
            
            # store on disk and close file
            ncfile[funcname] = value
            #print(f'Saving result to {filename}')
            ncfile.to_netcdf(filename)
            ncfile.close() # to prevent out of memory
        else:
            print('Warning: NaN or zero is detected in', instance.key+'-'+funcname)
            
        return value
        

def free_of_NaNs_and_zeros(value):
    '''
    This function checks that values in Xarray are not trivial 
    (no zeros, Nans and so on)
    '''
    if len(value.dims) <= 1:
        if value.notnull().sum() == value.size: # every scalar value is not NaN
            if (value==0.).sum() != value.size: # every scalar value is not 0.
                return True
            else:
                False
        else:
            False
    else:
        if value.notnull().sum() > value.size/2: # 50% of values are not nan
            return True
        else:
            False