## Jupyterlab extensions

```
jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager@2.0.0
jupyter labextension install --no-build @bokeh/jupyter_bokeh@2.0.4
jupyter labextension install --no-build @pyviz/jupyterlab_pyviz@1.0.4
jupyter labextension install --no-build bqplot@0.5.19
jupyter labextension install --no-build jupyter-matplotlib@0.7.4
jupyter labextension install --no-build jupyterlab-plotly@4.14.1
jupyter labextension install --no-build plotlywidget@4.14.1
jupyter lab build --dev-build=False --minimize=False
```

### Panos only

```
jupyter labextension install --no-build @axlair/jupyterlab_vim
jupyter labextension install --no-build jupyterlab_vim-system-clipboard-support
jupyter lab build --dev-build=False --minimize=False
```
