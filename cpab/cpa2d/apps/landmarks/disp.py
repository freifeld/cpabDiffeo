#!/usr/bin/env python
"""
Created on Thu Mar 24 12:41:52 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from pylab import plt
import of.plt


def disp(tw,theta,src,dst,transformed,level,use_subplots,
         scale_quiver):
    cpa_space = tw.ms.L_cpa_space[level] 

    markersize=2
    if use_subplots:
        fontsize=20
        fontsize_legend=15
    else:
        fontsize=30
        fontsize_legend=25
    if use_subplots:
        plt.figure(0)
        of.plt.set_figure_size_and_location(50,50,900,900)
            
    def fig_warp(h):
        plt.figure(h+1000)
    
    f=[fig_warp,plt.subplot][use_subplots]
    c=[plt.clf,plt.cla][use_subplots]

    h=331
    
    f(h)
    c()
    tw.disp_orig_grid_lines(level=-1,color='g')
    plt.title('grid')
   
#        plt.plot(src.cpu[:,0],src.cpu[:,1],'go',ms=markersize+1)
#            tw.config_plt(axis_on_or_off='on')
#        plt.title(r'$\mathrm{src}$',fontsize=fontsize)
    if any(tw.args.zero_v_across_bdry):
        tw.config_plt(axis_on_or_off='on')
    else:
        tw.config_plt(axis_on_or_off='on',use_lims=False) 
   

    h=332
    f(h)
    c()
    tw.disp_orig_grid_lines(level=-1,color='g')
  
    plt.plot(src.cpu[:,0],src.cpu[:,1],'go',ms=markersize+1)
    plt.title('grid+src')
    if any(tw.args.zero_v_across_bdry):
        tw.config_plt(axis_on_or_off='on')
    else:
        tw.config_plt(axis_on_or_off='on',use_lims=False) 
         
    
    h=333
    f(h)
    c()
    


    cpa_space.theta2Avees(theta)        
    tw.update_pat_from_Avees(level=tw.nLevels-1)
    
    
    plt.plot(dst.cpu[:,0],dst.cpu[:,1],'bo',ms=markersize+1)
    plt.title('dst')
    if any(tw.args.zero_v_across_bdry):
        tw.config_plt(axis_on_or_off='on')
    else:
        tw.config_plt(axis_on_or_off='on',use_lims=False) 
   
        
    
    h=334
    f(h)
    c()
    
   
    tw.quiver(scale=scale_quiver,
              ds=min([tw.nCols,tw.nRows])/32)
    
    if any(tw.args.zero_v_across_bdry):
        tw.config_plt(axis_on_or_off='on')
    else:
        tw.config_plt(axis_on_or_off='on',use_lims=False) 
   

    cpa_space.theta2Avees(theta)
    tw.update_pat_from_Avees(level=tw.nLevels-1)
    

    h=335        
    f(h)
    c()
    tw.imshow_vx()
    h=336        
    f(h)
    c()
    tw.imshow_vy()

 
    h=337        
    f(h)
    c()
    cpa_space.theta2Avees(theta)
    tw.update_pat_from_Avees(level=tw.nLevels-1)
    
    plt.plot(dst.cpu[:,0],dst.cpu[:,1],'bo',ms=markersize+1)

    plt.plot(src.cpu[:,0],src.cpu[:,1],'go',ms=markersize+1)
    if any(tw.args.zero_v_across_bdry):
        tw.config_plt(axis_on_or_off='on')
    else:
        tw.config_plt(axis_on_or_off='on',use_lims=False) 
   
    plt.title('src(g)+dst(b)')
    

    h=338       
    f(h)
    c()
    cpa_space.theta2Avees(theta)
    tw.update_pat_from_Avees(level=tw.nLevels-1)
    
    tw.disp_deformed_grid_lines(level=-1,color='r')
    plt.plot(transformed.cpu[:,0],transformed.cpu[:,1],'ro',ms=markersize+1)
    if any(tw.args.zero_v_across_bdry):
        tw.config_plt(axis_on_or_off='on')
    else:
        tw.config_plt(axis_on_or_off='on',use_lims=False) 
   
    plt.title('T(grid)+T(src)')
   
    h=339
    f(h)
    c()
    
    plt.plot(dst.cpu[:,0],dst.cpu[:,1],'bo',ms=markersize+1)
    plt.plot(transformed.cpu[:,0],transformed.cpu[:,1],'ro',ms=markersize+1)
    if any(tw.args.zero_v_across_bdry):
        tw.config_plt(axis_on_or_off='on')
    else:
        tw.config_plt(axis_on_or_off='on',use_lims=False)     

    plt.title('T(src)+dst')