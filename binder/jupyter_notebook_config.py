# 配置 Jupyter 环境默认以 Voila 模式打开您的笔记本
c.NotebookApp.default_url = '/voila/render/拒学信息提取-交互式.ipynb'

# 可选：配置 Voila 的一些行为
c.VoilaConfiguration.theme = 'dark'  # 使用深色主题，可选
c.VoilaConfiguration.enable_nbextensions = True  # 启用笔记本扩展

# 可选：隐藏代码单元格
c.VoilaConfiguration.strip_sources = True

# 可选：设置 Voila 模板
# c.VoilaConfiguration.template = 'gridstack'
