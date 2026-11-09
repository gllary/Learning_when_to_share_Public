"""Continuous-colour Supplementary Fig. 5, using full-data means.

The logarithmic mapping affects colour only. Each printed value is the
untransformed arithmetic mean of the five corresponding test-fold scores.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import NullLocator
import numpy as np
groups = ['composition_stoichiometry','electronic_chemistry','size_mass',
          'structure_geometry_density','symmetry','target_property']
labels = ['Composition\nand\nstoichiometry','Electronic and\nchemical\nproperties',
          'Atomic size\nand mass','Structural\ngeometry and\ndensity',
          'Crystal\nsymmetry','Target\nproperty']
STYLE = {
    'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','svg.hashsalt':'hpsafe-supp4-matrix-preview-20261108',
    'axes.unicode_minus':False,'text.color':'#283C48','axes.labelcolor':'#283C48',
    'xtick.color':'#283C48','ytick.color':'#283C48',
}
CMAP = LinearSegmentedColormap.from_list('HP_blue_continuous',
    ['#F1F6F8','#D8E9ED','#A6CAD5','#6AA6BB','#367F9B','#165471'],N=1024)
CMAP.set_bad('#E9EDF0')
NORM = LogNorm(vmin=1,vmax=20000,clip=False)
VALUE_COLOR = '#233B48'
NA_COLOR = '#89959C'

def base_layout(kind,tasks):
    fig=plt.figure(figsize=(9.6,7.7),facecolor='white')
    ax=fig.add_axes([.265,.25,.705,.61])
    ax.set_xlim(-.5,5.5);ax.set_ylim(12.5,-.5)
    ax.set_xticks(np.arange(6),labels,fontsize=8.6)
    ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=10)
    ax.set_yticks(np.arange(13),['matbench_'+x for x in tasks],fontsize=8.8)
    ax.tick_params(axis='y',length=0,pad=9)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.265,.975,'Representation alignment relative to a permutation reference',
             fontsize=11.3,va='top',weight='medium')
    # A subtle divider distinguishes the target column without changing its colour scale.
    ax.axvline(4.5,color='white' if kind=='heatmap' else '#D0DCE1',lw=3 if kind=='heatmap' else 1.0,zorder=3)
    return fig,ax

def colourbar(fig,y):
    cax=fig.add_axes([.33,y,.575,.022])
    m=matplotlib.cm.ScalarMappable(norm=NORM,cmap=CMAP)
    cb=fig.colorbar(m,cax=cax,orientation='horizontal',ticks=[1,10,100,1000,10000])
    cb.set_ticklabels(['1','10','100','1,000','10,000'])
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.ax.tick_params(labelsize=8,length=3,color='#9EB2BD',pad=4)
    cb.outline.set_linewidth(.6);cb.outline.set_edgecolor('#BACBD3')
    cb.set_label('Five-fold mean permutation z-score (logarithmic colour scale)',fontsize=9,labelpad=6)

def draw_heatmap(matrix,tasks):
    fig,ax=base_layout('heatmap',tasks)
    ax.pcolormesh(np.arange(7)-.5,np.arange(14)-.5,np.ma.masked_invalid(matrix),
                  cmap=CMAP,norm=NORM,edgecolors='white',linewidth=.8,shading='flat',zorder=1)
    ax.axvline(4.5,color='white',lw=3,zorder=3)
    for i in range(13):
        for j in range(6):
            v=matrix[i,j]
            if np.isfinite(v):
                # Match light/dark text to the actual mapped background.
                rgb=np.asarray(CMAP(NORM(v))[:3]);lum=float(np.dot(rgb,[.2126,.7152,.0722]))
                ax.text(j,i,f'{v:,.1f}',ha='center',va='center',fontsize=8.7,
                        color='white' if lum<.53 else VALUE_COLOR,zorder=4)
            else:
                ax.text(j,i,'N/A',ha='center',va='center',fontsize=8.3,color=NA_COLOR,fontstyle='italic',zorder=4)
    colourbar(fig,.145)
    fig.text(.265,.045,'Numbers show untransformed means; N/A indicates unavailable descriptor groups.',fontsize=8,color='#5E737E')
    return fig


def render_heatmap(summary,folds,tasks,output_stem,formats=('png','pdf','svg'),dpi=360):
    matrix=summary.pivot(index='task',columns='descriptor_group',values='permutation_z_mean').reindex(index=tasks,columns=groups).to_numpy()
    assert matrix.shape==(13,6)
    assert np.isfinite(matrix).sum()==70 and np.isnan(matrix).sum()==8
    for i,task in enumerate(tasks):
        for j,group in enumerate(groups):
            rows=folds[(folds.task==task)&(folds.descriptor_group==group)]
            assert len(rows)==5
            if np.isfinite(matrix[i,j]):
                assert (rows.status=='computed').all()
                np.testing.assert_allclose(matrix[i,j],rows.permutation_z.mean(),rtol=1e-12)
            else:
                assert (rows.status!='computed').all()
    assert np.nanmin(matrix)>=1 and np.nanmax(matrix)<=20000
    outputs={}
    with plt.rc_context({**matplotlib.rcParamsDefault,**STYLE}):
        fig=draw_heatmap(matrix,tasks)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for artist in fig.findobj(matplotlib.text.Text):
            if artist.get_visible() and artist.get_text():
                b=artist.get_window_extent(renderer)
                assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,artist.get_text()
        for axis in fig.axes:
            ticks=[t for t in axis.get_xticklabels() if t.get_visible() and t.get_text()]
            for left,right in zip(ticks,ticks[1:]):
                assert left.get_window_extent(renderer).x1+2<right.get_window_extent(renderer).x0
        for ext in ['png','pdf','svg']:
            if ext not in formats:continue
            path=Path(str(output_stem)+'.'+ext)
            path.parent.mkdir(parents=True,exist_ok=True)
            meta={'CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else {})
            fig.savefig(path,dpi=dpi,metadata=meta)
            outputs[ext]=path.name
        plt.close(fig)
    return {'display':'continuous-colour annotated heatmap','n_cells':78,'numeric_means':70,'na_entries':8,'mean_labels':[f'{v:,.1f}' for v in matrix.flat if np.isfinite(v)],'colour_scale':'logarithmic','colour_limits':[1,20000],'aggregation':'arithmetic mean before colour transformation','shows_individual_folds':False,'fold_results_location':'Supplementary Table 18','all_visible_text_inside_canvas':True,'outputs':outputs}
