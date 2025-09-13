# Algorithm Plugins

The algorithms package can be extended at runtime using plugins.  A plugin is a
Python package that exposes an entry point in the `autogpt.algorithms` group.

## Creating a plugin

1. Implement your algorithm in a module.
2. Add an entry point in your package's `pyproject.toml`:
   ```toml
   [project.entry-points."autogpt.algorithms"]
   my_plugin = "my_package.my_plugin"
   ```
3. Install the package so it is available on the Python path.

## Hot deployment

`AlgorithmPluginLoader` watches each loaded plugin's source file. When the file
changes the plugin module is automatically reloaded. If reloading fails the
previous working version is restored so your application keeps running.

```python
from algorithms import AlgorithmPluginLoader

loader = AlgorithmPluginLoader()
loader.load_plugins()
loader.start()
```

Stop the loader with `loader.stop()` when shutting down.
