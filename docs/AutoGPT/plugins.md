## Plugins

⚠️💀 **WARNING** 💀⚠️: Review the code of any plugin you use thoroughly, as plugins can execute any Python code, potentially leading to malicious activities, such as stealing your API keys.

To configure plugins, create or edit the `plugins_config.yaml` file in AutoGPT's `config` directory. This file allows you to enable or disable plugins as desired. For specific configuration instructions, please refer to the documentation provided for each plugin. The file should be formatted in YAML. Here is an example for your reference:

```yaml
plugin_a:
  config:
    api_key: my-api-key
  enabled: false
plugin_b:
  config: {}
  enabled: true
```

See our [Plugins Repo](https://github.com/Significant-Gravitas/Auto-GPT-Plugins) for more info on how to install all the amazing plugins the community has built!

Alternatively, developers can use the [AutoGPT Plugin Template](https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template) as a starting point for creating your own plugins.


### Runtime module management

AutoGPT now supports loading and unloading capability modules while the system
is running.  Modules are registered via ``module_registry`` and can be managed
through the :class:`RuntimeModuleManager`.

- **Add a module**

  ```http
  POST /modules {"module": "name"}
  ```
  or, in Python:
  ```python
  from capability.runtime_loader import RuntimeModuleManager
  RuntimeModuleManager().load("name")
  ```

- **Remove a module**

  ```http
  DELETE /modules/name
  ```
  or in Python:
  ```python
  RuntimeModuleManager().unload("name")
  ```

The manager keeps track of loaded modules and releases those no longer
required by incoming goals, allowing agents to extend their abilities
dynamically without a restart.

