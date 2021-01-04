## Jupyterlab extensions

```
jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager@2.0.0
jupyter labextension install --no-build jupyter-matplotlib@0.7.4
jupyter lab build --dev-build=False --minimize=False
```

### Panos only

```
jupyter labextension install --no-build @axlair/jupyterlab_vim
jupyter labextension install --no-build jupyterlab_vim-system-clipboard-support
jupyter lab build --dev-build=False --minimize=False
```
