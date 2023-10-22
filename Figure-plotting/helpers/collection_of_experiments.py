import xarray as xr
import os
from helpers.experiment import Experiment
from helpers.computational_tools import remesh, Lk_error
import cmocean
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class CollectionOfExperiments:
    '''
    This class extend dictionary of experiments by additional
    tools for plotting and comparing experiments
    '''
    def __init__(self, exps, experiments_dict, names_dict):
        '''
        experiments_dict - "experiment" objects labeled by keys
        names_dict - labels for plotting
        '''
        self.exps = exps
        self.experiments = experiments_dict
        self.names = names_dict

    def __getitem__(self, q):
        ''' 
        Access experiments with key values directly
        '''
        try:
            return self.experiments[q]
        except:
            print('item not found')
    
    def __add__(self, otherCollection):
        # merge dictionaries and lists
        exps = [*self.exps, *otherCollection.exps]
        experiments_dict = {**self.experiments, **otherCollection.experiments}
        names_dict = {**self.names, **otherCollection.names}

        return CollectionOfExperiments(exps, experiments_dict, names_dict)

    def compute_statistics(self, exps=None, recompute=False):
        if exps is None:
            exps = self.exps
        for exp in exps:
            if recompute:
                self[exp].recompute = True
            for key in Experiment.get_list_of_netcdf_properties():
                self[exp].__getattribute__(key)
            self[exp].recompute = False

    def remesh(self, input, target, exp=None, name=None, compute=False, operator=remesh):
        '''
        input  - key of experiment to coarsegrain
        target - key of experiment we want to take coordinates from
        '''

        if exp is None:
            exp = input+'_'+target
        if name is None:
            name = input+' coarsegrained to '+target

        result = self[input].remesh(self[target], exp, compute, operator) # call experiment method

        print('Experiment '+input+' coarsegrained to '+target+
            ' is created. Its identificator='+exp)
        self.exps.append(exp)
        self.experiments[exp] = result
        self.names[exp] = name
    
    @classmethod
    def init_folder(cls, common_folder, exps=None, exps_names=None, additional_subfolder='', prefix=None):
        '''
        Scan folders in common_folder and returns class instance with exps given by these folders
        exps - list of folders can be specified
        exps_names - list of labels can be specified
        additional_subfolder - if results are stored not in common_folder+exps[i],
        but in an additional subfolder 
        '''

        if exps is None:
            folders = sorted(os.listdir(common_folder))

        if exps_names is None:
            exps_names = folders

        if prefix:
            exps = [prefix+'-'+exp for exp in folders]
        else:
            exps = folders

        # Construct dictionary of experiments, where keys are given by exps
        experiments_dict = {}
        names_dict = {}
        for i in range(len(exps)):
            folder = os.path.join(common_folder,folders[i],additional_subfolder)
            experiments_dict[exps[i]] = Experiment(folder, exps[i])
            names_dict[exps[i]] = exps_names[i] # convert array to dictionary

        return cls(exps, experiments_dict, names_dict)
    
    def plot_KE_spectrum(self, exps, key='EKE_spectrum', labels=None):
        if labels is None:
            labels=exps
        fig, ax = plt.subplots(1,2,figsize=(15,6))
        p = []
        for j,exp in enumerate(exps):
            KE = self[exp].__getattribute__(key)
            k = KE.freq_r

            KE_upper = KE.isel(zl=0)
            KE_lower = KE.isel(zl=1)

            color = {exps[0]: 'gray', exps[-1]: 'k'}
            p.extend(ax[0].loglog(k, KE_upper, lw=3, label=labels[j], color=color.get(exp,None)))
            ax[0].set_xlabel(r'wavenumber, $k [m^{-1}]$')
            ax[0].set_ylabel(r'Energy spectrum, $E(k) [m^3/s^2]$')
            ax[0].set_title('Upper layer')
            ax[0].legend(prop={'size': 14})
            #ax[0].grid(which='both',linestyle=':')

            p.extend(ax[1].loglog(k, KE_lower, lw=3, label=labels[j], color=color.get(exp,None)))
            ax[1].set_xlabel(r'wavenumber, $k [m^{-1}]$')
            ax[1].set_ylabel(r'Energy spectrum, $E(k) [m^3/s^2]$')
            ax[1].set_title('Lower layer')
            ax[1].legend(prop={'size': 14})
            #ax[1].grid(which='both',linestyle=':')

        k = [5e-5, 1e-4]
        E = [1.5e+2, 0]
        E[1] = E[0] * (k[1]/k[0])**(-3)
        ax[0].loglog(k,E,'--k')
        ax[0].text(8e-5,5e+1,'$k^{-3}$')
        ax[0].set_xlim([None,1e-3])
        ax[0].set_ylim([1e-3,1e+3])
        
        ax[1].set_xlim([None,1e-3])
        ax[1].set_ylim([1e-3,1e+3])
        k = [5e-5, 1e-4]
        E = [3e+1, 0]
        E[1] = E[0] * (k[1]/k[0])**(-3)
        ax[1].loglog(k,E,'--k')
        ax[1].text(8e-5,1e+1,'$k^{-3}$')

        return p
    
    def plot_transfer(self, exp, target='R64_R4', callback=True):
        smag = self[exp].Smagorinsky_transfer
        ZB = self[exp].ZB_transfer
        kmax = self[exp].kmax
        if target is not None:
            SGS = self[target].SGS_transfer

        matplotlib.rcParams.update({'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm','axes.formatter.limits': (-1,2), 
        'axes.formatter.use_mathtext': True, 'font.size': 16})
        plt.figure(figsize=(15,4))
        for zl in range(2):
            plt.subplot(1,2,zl+1)
            if target is not None:
                SGS.isel(zl=zl).plot(label='SGS', color='k', ls='-')
            ZB.isel(zl=zl).plot(label='ZB', color='tab:orange', ls='--')
            smag.isel(zl=zl).plot(label='Smag', color='tab:green', ls='-.')
            (ZB+smag).isel(zl=zl).plot(label='ZB+Smag', color='tab:blue')
            plt.legend()
            plt.axhline(y=0,ls='-',color='gray',alpha=0.5)
            ax2 = plt.gca().secondary_xaxis('top', functions=(lambda x: x/kmax, lambda x: x*kmax))
            ax2.set_xlabel('Frequency/Nyquist')
            ax2.set_ticks([0.25, 0.5, 1],[r'$1/4$', r'$1/2$', r'$1$'])
            for k in [0.25, 0.5, 1]:
                plt.axvline(x=kmax*k,ls='-',color='gray',alpha=0.5)
            plt.xlim([None, kmax])
            plt.xlabel('wavenumber $k$ [m$^{-1}$]')
            plt.ylabel('KE transfer [m$^3$/s$^3$]')
            if zl==0:
                plt.title('Upper layer',fontweight='bold',fontsize=25, loc='right')
                plt.title('')
            else:
                plt.title('Lower layer',fontweight='bold',fontsize=25, loc='right')
                plt.title('')

        if callback:
            self.plot_power(exp,target)

    def plot_power(self, exp, target='R64_R4'):
        smag = self[exp].Smagorinsky_power
        ZB = self[exp].ZB_power
        model = self[exp].Model_power
        kmax = self[exp].kmax
        if target is not None:
            SGS = self[target].SGS_power

        matplotlib.rcParams.update({'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm','axes.formatter.limits': (-1,2), 
        'axes.formatter.use_mathtext': True, 'font.size': 16})
        plt.figure(figsize=(15,4))
        for zl in range(2):
            plt.subplot(1,2,zl+1)
            if target is not None:
                SGS.isel(zl=zl).plot(label='SGS', color='k', ls='-')
            ZB.isel(zl=zl).plot(label='ZB', color='tab:orange', ls='--')
            smag.isel(zl=zl).plot(label='Smag', color='tab:green', ls='-.')
            model.isel(zl=zl).plot(label='ZB+Smag', color='tab:blue')
            plt.legend()
            plt.axhline(y=0,ls='-',color='gray',alpha=0.5)
            for k in [0.25, 0.5, 1]:
                plt.axvline(x=kmax*k,ls='-',color='gray',alpha=0.5)
            plt.xlim([None, kmax])
            plt.xlabel('wavenumber $k$ [m$^{-1}$]')
            plt.ylabel('Power spectrum [m$^3$/s$^4$]')
            plt.title('')
            
    def plot_ssh(self, exps, labels=None, target=None):
        if labels is None:
            labels=exps
        plt.figure(figsize=(4*len(exps),3))
        nfig = len(exps)
        for ifig, exp in enumerate(exps):
            plt.subplot(1,nfig,ifig+1)
            if target is None or target == exp:
                ssh = self[exp].ssh_mean
                levels = np.arange(-4,4.5,0.5)
                label = 'SSH [m]'
                lines = True
            else:
                ssh = self[exp].ssh_mean
                ssh = ssh - remesh(self[target].ssh_mean,ssh)
                levels = np.linspace(-2,2,21)
                label = 'SSH error [m]'
                lines = False
            ssh.plot.contourf(levels=levels, cmap='bwr', linewidths=1, extend='both', cbar_kwargs={'label': label})
            if lines:
                Cplot = ssh.plot.contour(levels=levels, colors='k', linewidths=1)
                plt.gca().clabel(Cplot, Cplot.levels)
            plt.xticks((0, 5, 10, 15, 20))
            plt.yticks((30, 35, 40, 45, 50))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(labels[ifig])

            if exp != exps[-1]:
                RMSE = Lk_error(self[exp].ssh_mean,self[exps[-1]].ssh_mean)[0]
                #print(RMSE)
                plt.text(9,31,'RMSE='+str(round(RMSE,3))+'$m$', fontsize=14)

        plt.tight_layout()

    def plot_ssh_std(self, exps, labels=None, target='R64_R2'):
        if labels is None:
            labels=exps
        plt.figure(figsize=(4*len(exps),3))
        nfig = len(exps)
        for ifig, exp in enumerate(exps):
            plt.subplot(1,nfig,ifig+1)
            ssh = remesh(self[exp].ssh_std,self[target].ssh_std)
            levels = np.arange(0,1.3,0.1)
            label = 'SSH std [m]'

            ssh.plot.contourf(levels=levels, cmap=cmocean.cm.balance, linewidths=1, cbar_kwargs={'label': label})
            plt.xticks((0, 5, 10, 15, 20))
            plt.yticks((30, 35, 40, 45, 50))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(labels[ifig])

            if exp != exps[-1]:
                RMSE = Lk_error(ssh,remesh(self[exps[-1]].ssh_std, self[target].ssh_std))[0]
                plt.text(9,31,'RMSE='+str(round(RMSE,3))+'$m$', fontsize=14, color='w')

        plt.tight_layout()

    def plot_EKE(self, exps, labels=None, target='R64_R4', zl=0):
        if labels is None:
            labels=exps
        plt.figure(figsize=(4*len(exps),3))
        nfig = len(exps)
        for ifig, exp in enumerate(exps):
            plt.subplot(1,nfig,ifig+1)
            ssh = remesh(self[exp].EKE, self[target].EKE)
            if zl==0:
                levels = np.linspace(0,2.5e-2,11)
            else:
                levels = np.linspace(0,1.2e-2,13)
            label = 'EKE, $m^{2}s^{-2}$'

            ssh.isel(zl=zl).plot.contourf(levels=levels, cmap=cmocean.cm.balance, linewidths=1, cbar_kwargs={'label': label})
            plt.xticks((0, 5, 10, 15, 20))
            plt.yticks((30, 35, 40, 45, 50))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(labels[ifig])

            if exp != exps[-1]:
                RMSE = Lk_error(ssh,remesh(self[exps[-1]].EKE, self[target].EKE))[zl] 
                plt.text(2,31,'RMSE='+str(round(RMSE,5))+'$m^{2}s^{-2}$', fontsize=14, color='w')

        plt.tight_layout()

    def plot_RV(self, exps, labels=None,idx=-1, zl=0):
        if labels is None:
            labels=exps
        nfig = len(exps)
        ncol = min(3,nfig)
        nrows = nfig / 3
        if nrows > 1:
            nrows = int(np.ceil(nrows))
        else:
            nrows = 1
        plt.figure(figsize=(5*ncol,4*nrows))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        for ifig, exp in enumerate(exps):
            plt.subplot(nrows,ncol,ifig+1)
            field = self[exp].RV_f.isel(zl=zl,Time=idx)
            im = field.plot.imshow(vmin=-0.2, vmax=0.2, cmap=cmocean.cm.balance, 
                add_colorbar=False, interpolation='none')
            plt.xticks([0,5,10,15,20])
            plt.yticks([30,35,40,45,50])
            plt.xlim([0,22])
            plt.ylim([30,50])
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(labels[ifig])
        
        plt.colorbar(im, ax=plt.gcf().axes, label='Relative vorticity in \n Coriolis units, $\zeta/f$', extend='both')
        

    def plot_velocity(self, exps, labels=None, key='u_mean'):
        if labels is None:
            labels=exps
        plt.figure(figsize=(4*len(exps),3))
        nfig = len(exps)
        for ifig, exp in enumerate(exps):
            plt.subplot(1,nfig,ifig+1)
            v = self[exp].__getattribute__(key).isel(zl=0)
            levels = np.linspace(-0.3,0.3,21)
            label = 'Velocity [$m/s$]'
            v.plot.contourf(levels=levels, cmap='bwr', linewidths=1, extend='both', cbar_kwargs={'label': label})
            plt.xticks((0, 5, 10, 15, 20))
            plt.yticks((30, 35, 40, 45, 50))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(labels[ifig])

        plt.tight_layout()

    def plot_KE_PE(self, exps=['R4', 'R8', 'R64_R4'], labels=None, color=['k', 'tab:cyan', 'tab:blue', 'tab:red'], rotation=20):
        if labels is None:
            labels = exps
        plt.figure(figsize=(12,7))
        plt.subplots_adjust(wspace=0.3, hspace=0.7)
        width = (len(exps)-1) * [0.4] + [1]
        for zl in range(2):
            plt.subplot(2,2,zl+1)
            MKE = []
            EKE = []
            for exp in exps:          
                MKE.append(1e-15*self[exp].MKE_joul.isel(zl=zl).values)
                EKE.append(1e-15*self[exp].EKE_joul.isel(zl=zl).values)
            x=np.arange(len(exps));
            x[-1] += 1.5
            plt.bar(x,MKE,width,label='MKE',color=color[0])
            plt.bar(x,EKE,width,bottom=MKE,label='EKE',color=color[1])
            plt.ylabel('Kinetic energy, $PJ$');
            plt.xticks(ticks=x,labels=labels,rotation=rotation)
            if zl==0:
                plt.title('KE, Upper Layer')
            else:
                plt.title('KE, Lower Layer')
            plt.legend(loc='upper left', ncol=2)
            plt.axhline(y=MKE[-1], ls=':', color=color[0])
            if zl==0:
                plt.yticks([0,5,10,15,20,25,30])
            elif zl==1:
                plt.yticks([0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0])
            
        plt.subplot(2,2,3)
        MPE = []
        EPE = []
        for exp in exps:
            MPE.append(1e-15*(self[exp].MPE_joul.values+self[exp].MPE_ssh))
            EPE.append(1e-15*(self[exp].EPE_joul.values+self[exp].EPE_ssh))     
        x=np.arange(len(exps));
        x[-1] += 1.5
        plt.bar(x,MPE,width,label='MPE',color=color[2])
        plt.bar(x,EPE,width,bottom=MPE,label='EPE',color=color[3])
        plt.ylabel('Potential energy, $PJ$')
        plt.xticks(ticks=x,labels=labels,rotation=rotation)
        plt.title('Potential energy')
        plt.legend(loc='upper left', ncol=2)
        plt.ylim([0, (EPE[-1]+MPE[-1])*1.8])
        plt.yticks([0,25,50,75,100,125,150])
        plt.axhline(y=MPE[-1], ls=':', color=color[2])
        
        plt.subplot(2,2,4)
        EKE = []
        for exp in exps:          
            EKE.append(1e-15*self[exp].EKE_joul.values.sum())
        x=np.arange(len(exps));
        x[-1] += 1.5
        plt.bar(x,EKE,width,label='EKE',color=color[1])
        plt.bar(x,EPE,width,bottom=EKE, label='EPE',color=color[3])
        plt.ylabel('Eddy energy, $PJ$')
        plt.title('Energy of eddies')
        plt.xticks(ticks=x,labels=labels,rotation=rotation)
        plt.legend(loc='upper left', ncol=2)
        plt.ylim([0, 1.5*(EKE[-1]+EPE[-1])*1.4])
        plt.yticks([0,5,10,15,20,25,30,35,40])
        plt.axhline(y=EKE[-1], ls=':', color=color[1])

    def plot_KE_PE_simpler(self, exps=['R4', 'R8', 'R64_R4'], labels=None, color=['k', 'tab:cyan', 'tab:blue', 'tab:red'], rotation=20):
        if labels is None:
            labels = exps
        plt.figure(figsize=(10,6))
        plt.subplots_adjust(wspace=0.3, hspace=0.1)
        #width = (len(exps)-1) * [0.4] + [1]
        width = len(exps) * [0.4]
        
        plt.subplot(2,1,1)
        MKE = []
        EKE = []
        for exp in exps:          
            MKE.append(1e-15*self[exp].MKE_joul.sum('zl').values)
            EKE.append(1e-15*self[exp].EKE_joul.sum('zl').values)
        x=np.arange(len(exps));
        x[-1] += 0
        plt.bar(x,MKE,width,label='MKE',color=color[0])
        plt.bar(x,EKE,width,bottom=MKE,label='EKE',color=color[1])
        plt.ylabel('KE, \nJoules$\\times10^{15}$');
        plt.xticks(ticks=x,labels=[None]*len(x),rotation=rotation)
        #plt.title('Kinetic energy')
        plt.legend(loc='upper left', ncol=2)
        #plt.axhline(y=MKE[-1], ls=':', color=color[0])
        plt.axhline(y=MKE[-1]+EKE[-1], ls=':', color=color[1])
        plt.axhline(y=MKE[-2]+EKE[-2], ls=':', color=color[1])
        plt.axhspan(MKE[-2]+EKE[-2], MKE[-1]+EKE[-1], color=color[1], alpha=0.1, lw=0)
        plt.yticks([0,5,10,15,20,25,30])
            
        plt.subplot(2,1,2)
        MPE = []
        EPE = []
        for exp in exps:
            MPE.append(1e-15*(self[exp].MPE_joul.values+self[exp].MPE_ssh))
            EPE.append(1e-15*(self[exp].EPE_joul.values+self[exp].EPE_ssh))     
        x=np.arange(len(exps));
        x[-1] += 0
        plt.bar(x,MPE,width,label='MPE',color=color[2])
        plt.bar(x,EPE,width,bottom=MPE,label='EPE',color=color[3])
        plt.ylabel('APE, \nJoules$\\times10^{15}$')
        plt.xticks(ticks=x,labels=labels,rotation=rotation, fontsize=16)
        #plt.title('Potential energy')
        plt.legend(loc='upper left', ncol=2)
        #plt.ylim([0, (EPE[-1]+MPE[-1])*1.8])
        plt.ylim([0, 170])
        plt.yticks([0,25,50,75,100,125,150])
        #plt.axhline(y=MPE[-1], ls=':', color=color[2])
        plt.axhline(y=MPE[-1]+EPE[-1], ls=':', color=color[3])
        #plt.axhspan(MPE[-2]+EPE[-2], MPE[-1]+EPE[-1], color=color[3], alpha=1, lw=0)

    def plot_domain(self, axes=None):
        def plot(axes, xangle, yangle, topography, interface, free_surface, taux):
            topography = topography / 100
            interface = interface / 100
            free_surface = free_surface / 100

            # Manually set zorder to avoid bug in matplotlib
            # https://stackoverflow.com/questions/37611023/3d-parametric-curve-in-matplotlib-does-not-respect-zorder-workaround
            axes.computed_zorder = False

            [X,Y] = np.meshgrid(topography.xh, topography.yh)
            p1 = axes.plot_surface(X,Y,topography, label='Topography', edgecolor='none', zorder=-2, color=[0.8, 0.8, 0.8], alpha=1.0)
            p2 = axes.plot_surface(X,Y,interface, label='Interface', edgecolor='none', color='tab:orange', zorder=-2, alpha=0.5)
            # #p3 = axes.plot_surface(X,Y,topography*0-0.3, edgecolor='none', alpha=0.3)

            yy = taux.yh
            xx = np.ones_like(yy) * float(X.min())
            zz = np.ones_like(yy) * 1
            vector = taux[:,2]

            #axes.quiver(xx, yy, zz, vector, vector*0, vector*0, length = 100, alpha=1.0, linewidth=1, label='Wind stress', head_length=30)
            from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
            for x, y, z, v in zip(xx, yy, zz, vector):
                axes.add_artist(Line3D([x, x+v*100], [y, y], [z, z], color='tab:blue', linewidth=1.5, zorder=10))
                x = x+v*100 # arrow end
                arrow_size = 1
                arrow = [[
                    [x + arrow_size*0.5, y, z],
                    [x - arrow_size*0.5, y+arrow_size*0.3, z],
                    [x - arrow_size*0.5, y-arrow_size*0.3, z]
                ]]
                axes.add_collection3d(Poly3DCollection(arrow, color='tab:blue', zorder=10))

            [X,Y] = np.meshgrid(free_surface.xh, free_surface.yh)
            levels = 0.01*np.arange(-4,4.5,0.5)
            axes.contourf(X, Y, free_surface, levels=levels, cmap='bwr', zorder=-1, vmin=-0.04, vmax=0.04)
            axes.contour(X, Y, free_surface, levels=levels, colors='k', linewidths=1.5)
            
            axes.plot(np.nan,np.nan,'-',color='k',label='SSH contour')
            axes.plot(np.nan,np.nan,'->',color='tab:blue',label='Wind stress')

            axes.view_init(xangle, yangle)

            # https://stackoverflow.com/questions/55531760/is-there-a-way-to-label-multiple-3d-surfaces-in-matplotlib/55534939
            
            p1._facecolors2d = p1._facecolor3d
            p1._edgecolors2d = p1._facecolor3d

            p2._facecolors2d = p2._facecolor3d
            p2._edgecolors2d = p2._facecolor3d

            #p1.set_rasterized(True)
            #p2.set_rasterized(True)

            # p3._facecolors2d = p3._facecolor3d
            # p3._edgecolors2d = p3._facecolor3d
             
            axes.set_xlabel('Longitude', labelpad=5)
            axes.set_ylabel('Latitude', labelpad=5)
            axes.zaxis.set_rotate_label(False)
            axes.set_zlabel('Depth, $km$', labelpad=5, rotation=90)
            axes.set_yticks([30,35,40,45,50])
            axes.set_ylim([30,50])
            axes.set_xlim([0,22])
            axes.set_zticks([0, -5, -10, -15, -20],['$0.0$', '$0.5$', '$1.0$', '$1.5$', '$2.0$'], rotation=45)
            # axes.tick_params(axis='x', which='major', pad=1)
            # axes.tick_params(axis='y', which='major', pad=1)
            # axes.tick_params(axis='z', which='major', pad=1)
            axes.set_zlim([-20,2])
            axes.legend(fontsize=10, bbox_to_anchor=(0.3,0.3), loc='center')
        
        topography = self['R2'].e.isel(zi=2, Time=-1).coarsen(xh=2, yh=2, boundary='trim').mean()
        interface = xr.where(topography > -1000, np.nan, -1000)
        free_surface = self['R64_R4'].ssh_mean
        taux = self['R2'].forcing.taux.isel(Time=-1).coarsen(yh=2, boundary='trim').mean()

        if axes is None:
            axes = plt.gca(projection='3d')
        plot(axes, 30, 200, topography, interface, free_surface, taux)
        #plot(axes, 30, 230, topography, interface, free_surface, taux)